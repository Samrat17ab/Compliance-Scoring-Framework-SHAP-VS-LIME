"""Full paper-configuration C3b/C6 evidence; no explainer sensitivity tuning.

Run: .venv/Scripts/python.exe -u paper_c3b_c6_evidence.py
Render saved results: add --render-only. Resume uses saved sequential RNG state.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import hashlib
import importlib.metadata
import json
import pickle
import sys
import time
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.special import expit
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/paper_c3b_c6_evidence'
OUT.mkdir(parents=True,exist_ok=True)
N,K,SEED=1000,10,42
CATEGORIES=['document_flags','encoded_categoricals','opaque_scores','plain_meaning','other']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def category(name,encoded):
    if name.startswith('FLAG_DOCUMENT_'): return 'document_flags'
    if name in encoded: return 'encoded_categoricals'
    if name.startswith('EXT_SOURCE_'): return 'opaque_scores'
    if name.startswith(('AMT_','CNT_')) or '_CNT_' in name or name in ['DAYS_BIRTH','DAYS_EMPLOYED','OWN_CAR_AGE']:
        return 'plain_meaning'
    return 'other'

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')

def compute():
    np.random.seed(SEED)
    start=time.time()
    df=pd.read_csv(ROOT/'application_train.csv')
    y=df['TARGET']
    X=df.drop(columns=['TARGET','SK_ID_CURR'])
    X=X.drop(columns=X.columns[X.isna().all()])
    encoded=X.select_dtypes(include='object').columns.tolist()
    mappings={}
    for col in encoded:
        le=LabelEncoder()
        X[col]=le.fit_transform(X[col].astype(str))
        mappings[col]=[None if pd.isna(v) else str(v) for v in le.classes_]
    xt,xte,yt,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    xtr,xva,ytr,yva=train_test_split(xt,yt,test_size=.2,random_state=42,stratify=yt)
    medians=xtr.median(numeric_only=True)
    train=xtr.fillna(medians)
    raw=xte.sample(N,random_state=42)
    sample=raw.fillna(medians)
    names=sample.columns.tolist()
    assert not sample.isna().any().any()
    previous=json.loads((ROOT/'output/applicant_comparisons/explanation_data.json').read_text())
    model_path=ROOT/'output/applicant_comparisons/retrained_lightgbm.txt'
    assert sha(model_path)==previous['model_sha256']
    assert sha(ROOT/'application_train.csv')==previous['dataset_sha256']
    assert names==previous['feature_names']
    assert np.allclose(medians.to_numpy(),[previous['training_medians'][n] for n in names])
    model=lgb.Booster(model_file=str(model_path))
    model.save_model(str(OUT/'lightgbm_model.txt'))
    probs=model.predict(sample,num_threads=4)
    raw_scores=model.predict(raw,num_threads=4)
    margins=model.predict(sample,raw_score=True,num_threads=4)
    print(f'TreeSHAP: all {N} shared median-filled inputs...',flush=True)
    tree=shap.TreeExplainer(model)
    vals=np.asarray(tree.shap_values(sample))
    assert vals.shape==(N,len(names))
    base=float(np.asarray(tree.expected_value).reshape(-1)[0])
    residual=np.abs(base+vals.sum(axis=1)-margins)
    assert residual.max()<1e-7
    assert np.allclose(expit(margins),probs,atol=1e-12)
    top=np.argsort(np.abs(vals),axis=1)[:,::-1][:,:K]
    cats={n:category(n,encoded) for n in names}

    # Match original main-analysis construction and call order exactly.
    lime=LimeTabularExplainer(training_data=train.values,feature_names=names,
             class_names=['Repaid','Default'],mode='classification',
             discretize_continuous=True,random_state=42)
    def predict_fn(a):
        p=model.predict(pd.DataFrame(a,columns=names),num_threads=4)
        return np.column_stack([1-p,p])
    cache=OUT/'lime_sequential_checkpoint.pkl'
    signature=dict(model=sha(model_path),sample_ids=df.loc[sample.index,'SK_ID_CURR'].tolist(),
                   versions={p:importlib.metadata.version(p) for p in ['lime','numpy','scikit-learn']})
    records=[]
    if cache.exists():
        # This file is generated locally by this script, never from an upload.
        with cache.open('rb') as f: saved=pickle.load(f)
        assert saved['signature']==signature
        records=saved['records']
        lime.random_state.set_state(saved['rng_state'])
        assert lime.base.random_state is lime.random_state
        assert lime.discretizer.random_state is lime.random_state
        print(f'Resuming at {len(records)} with exact sequential RNG state.',flush=True)
    for i in range(len(records),N):
        with threadpool_limits(limits=1,user_api='blas'):
            exp=lime.explain_instance(data_row=sample.iloc[i].values,predict_fn=predict_fn,
                                     num_features=10,num_samples=1000)
        pairs=exp.as_list(label=1)
        indices=[int(idx) for idx,v in exp.local_exp[1]]
        weights=[float(v) for idx,v in exp.local_exp[1]]
        assert len(indices)==10 and len(set(indices))==10
        assert abs(exp.predict_proba[1]-probs[i])<1e-12
        assert np.allclose(weights,[v for s,v in pairs])
        local=float(exp.local_pred[0])
        intercept=float(exp.intercept[1])
        assert abs(intercept+sum(weights)-local)<1e-7
        terms=[]
        for idx,(condition,w) in zip(indices,pairs):
            terms.append(dict(feature_index=idx,feature=names[idx],condition=condition,
                              weight=float(w),category=cats[names[idx]],
                              is_bin_or_range=('<' in condition or '>' in condition)))
        records.append(dict(sample_position=i,applicant_id=int(df.loc[sample.index[i],'SK_ID_CURR']),
                    r2=float(exp.score),surrogate_prediction=local,intercept=intercept,
                    absolute_surrogate_gap=abs(local-float(probs[i])),top10=terms))
        if (i+1)%25==0 or i==N-1:
            with cache.open('wb') as f:
                pickle.dump(dict(signature=signature,records=records,rng_state=lime.random_state.get_state()),f)
            print(f'LIME {i+1}/{N} | elapsed {(time.time()-start)/60:.1f} min',flush=True)
    per=[]
    for i,r in enumerate(records):
        st=[dict(feature_index=int(j),feature=names[j],value=float(sample.iloc[i,j]),
                 imputed=bool(pd.isna(raw.iloc[i,j])),phi=float(vals[i,j]),category=cats[names[j]]) for j in top[i]]
        per.append(dict(sample_position=i,applicant_id=r['applicant_id'],source_row_index=int(sample.index[i]),
                  score=float(probs[i]),original_raw_input_score=float(raw_scores[i]),
                  decision='rejected' if probs[i]>=.5 else 'accepted',
                  shap_base=base,model_log_odds=float(margins[i]),
                  shap_sum_all=float(vals[i].sum()),shap_reconstruction_error=float(residual[i]),
                  shap_top10=st,lime=r,
                  shap_category_counts={c:sum(t['category']==c for t in st) for c in CATEGORIES},
                  lime_category_counts={c:sum(t['category']==c for t in r['top10']) for c in CATEGORIES}))
    gaps=np.array([r['absolute_surrogate_gap'] for r in records])
    part_a=dict(traceability=dict(
        shap_reconstruction_error_max=float(residual.max()),shap_units='model log-odds',
        lime_absolute_surrogate_gap_mean=float(gaps.mean()),
        lime_absolute_surrogate_gap_median=float(np.median(gaps)),
        lime_absolute_surrogate_gap_p90=float(np.percentile(gaps,90)),
        lime_units='model probability / uncalibrated class-score units',percentile_method='linear',
        interpretation='Different checks in different units; their magnitudes must not be compared as a ratio.'),
        representation=dict(
            lime_bin_or_range_item_count=sum(t['is_bin_or_range'] for r in records for t in r['top10']),
            lime_bin_or_range_item_pct=100*np.mean([t['is_bin_or_range'] for r in records for t in r['top10']]),
            shap_bin_or_range_item_pct=0.0,shap_exact_model_input_value_display_pct=100.0,
            denominator_items=N*K,
            interpretation='Representation proxy only. SHAP labels attach the actual model input; SHAP itself returns attributions. Binary-feature bins may identify a single value. Neither method establishes causal actionability.'))
    composition={}
    for method in ['shap','lime']:
        composition[method]={}
        for cat in CATEGORIES:
            counts=[p[method+'_category_counts'][cat] for p in per]
            composition[method][cat]=dict(total_items=int(sum(counts)),
                 mean_items_per_top10=float(np.mean(counts)),mean_pct_of_top10=float(np.mean(counts)*10))
    burden={}
    gender={}
    for method in ['shap','lime']:
        counts=[p[method+'_category_counts']['document_flags']+p[method+'_category_counts']['encoded_categoricals'] for p in per]
        burden[method]=dict(instance_count=int(np.sum(np.array(counts)>=5)),pct_instances=float(100*np.mean(np.array(counts)>=5)))
        lists=[p['shap_top10'] if method=='shap' else p['lime']['top10'] for p in per]
        count=sum(any(t['feature']=='CODE_GENDER' for t in ls) for ls in lists)
        gender[method]=dict(instance_count=count,pct_instances=100*count/N)
    part_b=dict(top10_composition=composition,at_least_5_document_flags_or_encoded_categoricals=burden,
                code_gender_in_top10=gender,label_encoded_columns=encoded,
                category_definitions=dict(document_flags='FLAG_DOCUMENT_*',encoded_categoricals='All columns label-encoded from object/string data before splitting.',
                    opaque_scores='EXT_SOURCE_*',plain_meaning='AMT_* or CNT_* or names containing _CNT_; plus DAYS_BIRTH, DAYS_EMPLOYED, OWN_CAR_AGE.',
                    other='All remaining fields, including other day offsets, regional ratings, housing aggregates, and other flags. No column is silently assigned to plain meaning.'),
                feature_category_map=cats,
                interpretation='Prespecified name-based content proxy; not measured human comprehension. Plain meaning does not guarantee reader understanding. Categories are disjoint and exhaustive.')
    selected={}
    for decision in ['accepted','rejected']:
        group=[p for p in per if p['decision']==decision]
        med=float(np.median([p['lime']['r2'] for p in group]))
        chosen=min(group,key=lambda p:(abs(p['lime']['r2']-med),p['sample_position']))
        selected[decision]=dict(applicant_id=chosen['applicant_id'],sample_position=chosen['sample_position'],
                group_count=len(group),group_median_lime_r2=med,selected_lime_r2=chosen['lime']['r2'],
                absolute_distance_to_group_median=abs(chosen['lime']['r2']-med),score=chosen['score'])
    result=dict(n_instances=N,top_k_for_statistics=10,display_top_k=8,
          configuration=dict(status='Paper explainer configuration; user-requested shared-input rerun. No sensitivity variants.',
             sensitivity_variants=[],input_policy='Both explainers receive identical training-median-filled inputs; original paper code gave SHAP raw missing values.',
             model_provenance='Existing retrained paper-settings LightGBM from previous task; original fitted paper model unavailable. 190 rounds versus 204 in original console output.',
             model_params=previous['model_params'],model_iterations=model.current_iteration(),
             dataset_sha256=previous['dataset_sha256'],model_sha256=sha(model_path),
             source_code_sha256=sha(ROOT/'csf_evaluation.py'),analysis_code_sha256=sha(Path(__file__)),
             split_sizes=[len(xtr),len(xva),len(xte)],sample_random_state=42,
             shap='shap.TreeExplainer(model), defaults; raw/log-odds output, tree-path-dependent without background data',
             lime=dict(discretize_continuous=True,num_features=10,num_samples=1000,categorical_features='not passed',
                random_state=42,seed_policy='Single explainer; sequential calls in sampled row order; no per-instance reseeding.',
                kernel_width=.75*np.sqrt(len(names)),discretizer='quartile',feature_selection='auto',mode='classification'),
             versions={p:importlib.metadata.version(p) for p in ['lightgbm','shap','lime','pandas','numpy','scikit-learn','matplotlib']}),
          part_a=part_a,part_b=part_b,part_c=dict(rule='Within each predicted class (score < 0.5 accepted; >= 0.5 rejected), select the applicant minimizing absolute distance from that class median LIME R²; ties resolved by earliest position in the original seed-42 sample. No further exclusions.',selected=selected),
          sample_applicant_ids=[p['applicant_id'] for p in per],training_medians=medians.to_dict(),label_encoding_maps=mappings,
          per_instance=per)
    write_json(OUT/'evidence.json',result)
    np.savez_compressed(OUT/'full_shap_and_inputs.npz',shap_values=vals,inputs=sample.to_numpy(),
                        feature_names=np.array(names),applicant_ids=np.array(result['sample_applicant_ids']))
    print(json.dumps(dict(part_a=part_a,composition=composition,burden=burden,gender=gender,part_c=result['part_c']),indent=2),flush=True)
    return result

INK='#15283d'; MUTED='#506176'; SHAP='#28639a'; LIME='#b96820'; POS='#bd443e'; NEG='#247a9e'
# 13 1/3 inches is the exact 16:9 width, conventionally written 13.33 inches.
W,H,DPI=40/3,7.5,240
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':INK,'svg.fonttype':'none'})

def new_slide(title,subtitle):
    fig=plt.figure(figsize=(W,H),facecolor='white')
    fig.add_artist(Rectangle((0,.86),1,.14,transform=fig.transFigure,facecolor=INK,edgecolor='none',zorder=-1))
    fig.text(.03,.94,title,fontsize=24,color='white',weight='bold',va='center')
    fig.text(.03,.885,subtitle,fontsize=12.5,color='#e0e8f0',va='center')
    return fig

def footer(fig,summary=False):
    fig.text(.03,.065,'Same training-median-filled input for both methods. Panel units differ: SHAP log-odds; LIME probability.',fontsize=10.3,color=MUTED)
    fig.text(.03,.038,'Score is uncalibrated. Condition bins are not approval thresholds. Paper explainer settings; shared-input rerun.',fontsize=10.3,color=MUTED)
    if not summary:
        fig.text(.03,.092,'Top 8 displayed from the top-10 explanation; omitted terms retained in reconstruction. * Median-filled value.',fontsize=10.3,color=MUTED)

def exact(v):
    return str(int(v)) if float(v).is_integer() else repr(float(v))

def callout(fig,num,x,y,text,width):
    fig.add_artist(Circle((x+.010,y+.003),.0105,transform=fig.transFigure,facecolor=INK,edgecolor='none'))
    fig.text(x+.010,y+.003,str(num),fontsize=10.5,color='white',ha='center',va='center',weight='bold')
    fig.text(x+.027,y+.003,text,fontsize=11.2,va='center',linespacing=1.2)

def example_slide(data,decision):
    sel=data['part_c']['selected'][decision]
    p=data['per_instance'][sel['sample_position']]
    l=p['lime']
    fig=new_slide(f"{decision.upper()}  |  Applicant {p['applicant_id']}  |  Score {p['score']:.3f}",
       f"Chosen nearest the {decision}-group median LIME R²: {sel['group_median_lime_r2']:.3f}  |  Selected R² {l['r2']:.3f}")
    for x,title,col in [(.03,'SHAP',SHAP),(.535,'LIME',LIME)]:
        fig.text(x,.813,title,fontsize=21,weight='bold',color=col)
    fig.text(.03,.777,'Actual feature value + attribution',fontsize=12.5,color=MUTED)
    fig.text(.535,.777,'Condition + surrogate coefficient',fontsize=12.5,color=MUTED)
    fig.add_artist(plt.Line2D([.505,.505],[.275,.825],transform=fig.transFigure,color='#dce4ec'))
    for side in [0,1]:
        terms=p['shap_top10'][:8] if side==0 else l['top10'][:8]
        if side==0:
            labels=[t['feature']+'\n= '+exact(t['value'])+(' *' if t['imputed'] else '') for t in terms]
            values=[t['phi'] for t in terms]
        else:
            labels=[textwrap.fill(t['condition'],width=29,break_long_words=False,break_on_hyphens=False) for t in terms]
            values=[t['weight'] for t in terms]
        # One negative/positive scale per panel; units explicitly differ.
        ax=fig.add_axes([.31+side*.50,.31,.135,.46])
        ax.barh(np.arange(8),values,height=.50,color=[POS if v>0 else NEG for v in values],zorder=3)
        ax.set_yticks(np.arange(8),labels,fontsize=14)
        ax.tick_params(axis='y',length=0,pad=8)
        for tick in ax.get_yticklabels(): tick.set_linespacing(1.05)
        ax.set_ylim(7.65,-.65)
        limit=max(abs(v) for v in values)*1.10
        ax.set_xlim(-limit,limit)
        ax.axvline(0,lw=.8,color=MUTED)
        ax.grid(axis='x',color='#e7edf3',zorder=0)
        ax.tick_params(axis='x',labelsize=10)
        ax.locator_params(axis='x',nbins=3)
        for spine in ax.spines.values(): spine.set_visible(False)
        for i,v in enumerate(values):
            ax.text(1.06,i,f'{v:+.3f}',transform=ax.get_yaxis_transform(),fontsize=10.5,va='center',ha='left')
        ax.set_xlabel('Log-odds contribution' if side==0 else 'Probability-scale weight',fontsize=11,labelpad=5)
    fig.add_artist(Rectangle((.02,.125),.96,.135,transform=fig.transFigure,facecolor='#f1f5f8',edgecolor='none',zorder=-1))
    callout(fig,1,.031,.232,'Direction: red raises default score; blue lowers it.',.47)
    callout(fig,2,.535,.232,'Magnitude: signed numbers; different panel units.',.45)
    callout(fig,3,.031,.172,'Value vs bin: SHAP input values; LIME conditions.\nNeither establishes a causal intervention.',.47)
    callout(fig,4,.535,.172,f"Reconstruction error: {p['shap_reconstruction_error']:.1e} log-odds\nSurrogate gap: {l['absolute_surrogate_gap']:.3f} probability units",.45)
    footer(fig)
    return fig

def summary_slide(data):
    fig=new_slide('C3b + C6  |  Evidence across 1,000 applicants',
                  'Statistics use all top-10 items per applicant (10,000 items per method). No explainer tuning.')
    fig.text(.03,.805,'C3b  |  Top-10 content',fontsize=18,weight='bold')
    fig.text(.56,.805,'C6  |  Representation + traceability',fontsize=18,weight='bold')
    # Left: mutually exclusive categories; mean of per-instance percentages.
    cats=CATEGORIES
    labels=['Document flags','Encoded categories','Opaque scores','Plain-meaning fields','Other fields']
    ax=fig.add_axes([.20,.43,.285,.325])
    s=[data['part_b']['top10_composition']['shap'][c]['mean_pct_of_top10'] for c in cats]
    l=[data['part_b']['top10_composition']['lime'][c]['mean_pct_of_top10'] for c in cats]
    y=np.arange(5)
    for vals,offset,col in [(s,-.17,SHAP),(l,.17,LIME)]:
        ax.barh(y+offset,vals,height=.30,color=col)
        for yy,v in zip(y+offset,vals): ax.text(v+.8,yy,f'{v:.2f}%',fontsize=11,va='center')
    ax.set_yticks(y,labels,fontsize=14)
    ax.set_ylim(4.6,-.6); ax.set_xlim(0,65)
    ax.set_xlabel('Mean share of top-10 items (%)',fontsize=12)
    ax.tick_params(axis='both',length=0)
    ax.tick_params(axis='x',labelsize=11)
    ax.spines[['top','right','left']].set_visible(False)
    # Right: percentages with explicit denominator descriptions.
    ar=data['part_a']['representation']; b=data['part_b']
    pairs=[('Items shown as bins',ar['shap_bin_or_range_item_pct'],ar['lime_bin_or_range_item_pct']),
           ('≥5 flags / encoded¹',b['at_least_5_document_flags_or_encoded_categoricals']['shap']['pct_instances'],b['at_least_5_document_flags_or_encoded_categoricals']['lime']['pct_instances']),
           ('Gender in top 10¹',b['code_gender_in_top10']['shap']['pct_instances'],b['code_gender_in_top10']['lime']['pct_instances'])]
    ax2=fig.add_axes([.765,.475,.17,.245])
    for yy,(label,sv,lv) in enumerate(pairs):
        for offset,v,col in [(-.17,sv,SHAP),(.17,lv,LIME)]:
            ax2.barh(yy+offset,v,height=.3,color=col)
            ax2.text(v+2,yy+offset,f'{v:.2f}%',fontsize=11,va='center')
    ax2.set_yticks(range(3),[p[0] for p in pairs],fontsize=12.5)
    ax2.set_ylim(2.6,-.6); ax2.set_xlim(0,125)
    ax2.set_xticks([0,50,100]); ax2.tick_params(axis='x',labelsize=10)
    ax2.tick_params(axis='both',length=0); ax2.spines[['top','right','left']].set_visible(False)
    fig.text(.56,.422,'¹ % of instances; first row = % of items.',fontsize=11,color=MUTED)
    t=data['part_a']['traceability']
    fig.add_artist(Rectangle((.55,.265),.42,.12,transform=fig.transFigure,facecolor='#f1f5f8',edgecolor='none',zorder=-1))
    fig.text(.563,.355,f"SHAP max reconstruction error: {t['shap_reconstruction_error_max']:.2e}",fontsize=12,weight='bold')
    fig.text(.563,.325,'LIME gap: mean / median / 90th percentile',fontsize=12)
    fig.text(.563,.291,f"{t['lime_absolute_surrogate_gap_mean']:.3f}  /  {t['lime_absolute_surrogate_gap_median']:.3f}  /  {t['lime_absolute_surrogate_gap_p90']:.3f} probability units",fontsize=14,weight='bold')
    fig.text(.03,.326,'SHAP',fontsize=13,color=SHAP,weight='bold')
    fig.text(.12,.326,'LIME',fontsize=13,color=LIME,weight='bold')
    fig.text(.03,.285,'Exact model-input values attached to SHAP labels: 100%.',fontsize=11.5)
    fig.text(.03,.217,'Content and representation are proxies, not validated comprehension or causal actionability scores.',fontsize=12,weight='bold')
    fig.text(.03,.175,'Plain meaning: amounts, counts, age / employment days, car age. Other fields remain separate.',fontsize=11.5,color=MUTED)
    fig.text(.03,.136,'SHAP reconstruction uses log-odds; LIME gap uses probability. Errors are not directly comparable.',fontsize=11.5,color=MUTED)
    footer(fig,True)
    return fig

def render(data):
    for decision in ['accepted','rejected']:
        fig=example_slide(data,decision)
        fig.savefig(OUT/f"{decision}_example.png",dpi=DPI)
        plt.close(fig)
    fig=summary_slide(data)
    fig.savefig(OUT/'summary_1000.png',dpi=DPI)
    plt.close(fig)
    print('Rendered three 3200 x 1800 PNGs.',flush=True)

if __name__=='__main__':
    d=json.loads((OUT/'evidence.json').read_text()) if '--render-only' in sys.argv else compute()
    render(d)
