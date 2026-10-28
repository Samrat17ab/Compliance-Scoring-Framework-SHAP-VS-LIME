"""Recompute four paired explanations using the paper's LightGBM settings.

Run from the project root with .venv/Scripts/python.exe. All outputs are new;
the paper, original scripts, and published aggregate figures are untouched.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import copy
import hashlib
import importlib.metadata
import json
from itertools import combinations
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output' / 'applicant_comparisons'
OUT.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings('ignore', message='LightGBM binary classifier')
SEED, K, NS = 42, 10, 1000

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def compute():
    np.random.seed(SEED)
    print('Loading original dataset and recreating paper splits...', flush=True)
    df = pd.read_csv(ROOT / 'application_train.csv')
    y = df.TARGET
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    X = X.drop(columns=X.columns[X.isna().all()])
    mappings = {}
    for col in X.select_dtypes(include='object'):
        enc = LabelEncoder()
        X[col] = enc.fit_transform(X[col].astype(str))
        mappings[col] = [None if pd.isna(v) else str(v) for v in enc.classes_]
    xt, xte, yt, yte = train_test_split(X, y, test_size=.2, random_state=SEED, stratify=y)
    xtr, xva, ytr, yva = train_test_split(xt, yt, test_size=.2, random_state=SEED, stratify=yt)
    medians = xtr.median(numeric_only=True)
    train_filled = xtr.fillna(medians)
    names = X.columns.tolist()
    params = dict(objective='binary', metric='auc', boosting_type='gbdt',
                  num_leaves=31, learning_rate=.05, feature_fraction=.9,
                  bagging_fraction=.8, bagging_freq=5,
                  scale_pos_weight=float((ytr == 0).sum() / (ytr == 1).sum()),
                  verbose=-1, random_state=SEED, num_threads=4)
    model_path = OUT / 'retrained_lightgbm.txt'
    if model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        print('Loading cached model from this figure-generation run.', flush=True)
    else:
        print('Training LightGBM with original settings; thread count capped at 4.', flush=True)
        model = lgb.train(params, lgb.Dataset(xtr, label=ytr), num_boost_round=1000,
                          valid_sets=[lgb.Dataset(xva, label=yva)],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
        model.save_model(str(model_path))
    auc = roc_auc_score(yte, model.predict(xte, num_threads=4))
    print(f'Test AUC on original raw-input pipeline: {auc:.6f}', flush=True)
    raw = xte.sample(1000, random_state=SEED)
    filled = raw.fillna(medians)
    ps = model.predict(filled, num_threads=4)
    raw_ps = model.predict(raw, num_threads=4)
    stable = (ps >= .5) == (raw_ps >= .5)
    # Select by model decisions only, never by explanation metrics or labels.
    positions = list(np.flatnonzero((ps < .5) & stable)[:2]) + list(np.flatnonzero((ps >= .5) & stable)[:2])
    rows = filled.iloc[positions]
    raw_rows = raw.iloc[positions]
    explainer = shap.TreeExplainer(model, model_output='raw', feature_perturbation='tree_path_dependent')
    sv = np.asarray(explainer.shap_values(rows))
    baseline = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    margins = model.predict(rows, raw_score=True, num_threads=4)
    assert np.allclose(baseline + sv.sum(axis=1), margins, atol=1e-7)
    assert np.allclose(expit(margins), ps[positions], atol=1e-10)
    shap_repeat = np.asarray(explainer.shap_values(rows))
    assert np.array_equal(sv, shap_repeat)

    def predict_fn(a):
        p = model.predict(pd.DataFrame(a, columns=names), num_threads=4)
        return np.column_stack([1-p, p])

    cases = []
    for j, pos in enumerate(positions):
        idx = int(rows.index[j])
        top = np.argsort(np.abs(sv[j]))[::-1][:K]
        decision = 'accepted' if ps[pos] < .5 else 'rejected'
        cases.append(dict(case=f'{decision}_{j % 2 + 1}', decision=decision,
                          applicant_id=int(df.loc[idx, 'SK_ID_CURR']),
                          source_row_index=idx, csv_line=idx+2, sample_position=int(pos),
                          observed_target=int(y.loc[idx]), score=float(ps[pos]),
                          original_raw_score=float(model.predict(raw_rows.iloc[[j]], num_threads=4)[0]),
                          imputed_features=raw_rows.columns[raw_rows.iloc[j].isna()].tolist(),
                          input=rows.iloc[j].to_dict(), raw_input=raw_rows.iloc[j].astype(object).where(raw_rows.iloc[j].notna(), None).to_dict(),
                          shap_values=sv[j].tolist(), shap_base=baseline,
                          shap_margin=float(margins[j]), shap_top_indices=top.tolist(),
                          shap_other_sum=float(sv[j].sum()-sv[j, top].sum()),
                          shap_top10_coverage=float(np.abs(sv[j, top]).sum()/np.abs(sv[j]).sum()),
                          shap_repeat_max_abs_delta=float(np.max(np.abs(sv[j]-shap_repeat[j]))),
                          lime_runs=[]))
    for seed in range(42, 52):
        print(f'LIME: seed {seed}, four applicants, {NS} perturbations each...', flush=True)
        # A pristine post-construction explainer is copied per applicant so call
        # order cannot change its seed state. This is a fresh per-instance run.
        template = LimeTabularExplainer(train_filled.values, feature_names=names,
                    class_names=['Repaid', 'Default'], mode='classification',
                    discretize_continuous=True, random_state=seed)
        for j, case in enumerate(cases):
            le = copy.deepcopy(template)
            with threadpool_limits(limits=1, user_api='blas'):
                e = le.explain_instance(rows.iloc[j].values, predict_fn,
                                       labels=(1,), num_features=K, num_samples=NS)
            assert abs(e.predict_proba[1]-case['score']) < 1e-10
            terms = e.as_list(label=1)
            run = dict(seed=seed, r2=float(e.score), intercept=float(e.intercept[1]),
                       local_prediction=float(e.local_pred[0]),
                       indices=[int(i) for i, v in e.local_exp[1]],
                       terms=[{'condition': n, 'weight': float(v)} for n,v in terms])
            assert abs(run['intercept']+sum(t['weight'] for t in run['terms'])-run['local_prediction']) < 1e-7
            case['lime_runs'].append(run)
            if seed == 42:
                e.save_to_file(str(OUT / f"{case['case']}_lime_native.html"))
                with threadpool_limits(limits=1, user_api='blas'):
                    repeat = copy.deepcopy(template).explain_instance(rows.iloc[j].values, predict_fn,
                                      labels=(1,), num_features=K, num_samples=NS)
                assert e.as_list(label=1) == repeat.as_list(label=1)
                case['lime_fixed_seed_repeat_max_abs_delta'] = float(np.max(np.abs(
                    np.array([v for _,v in e.local_exp[1]]) - np.array([v for _,v in repeat.local_exp[1]]))))
    for c in cases:
        sets = [set(r['indices']) for r in c['lime_runs']]
        c['lime_10_seed_mean_pairwise_jaccard'] = float(np.mean([len(a&b)/len(a|b) for a,b in combinations(sets,2)]))
    results = dict(feature_names=names, cases=cases, model_params=params,
                   num_boost_round=1000, early_stopping_rounds=50,
                   trained_iterations=model.current_iteration(), raw_test_auc=float(auc),
                   split_sizes=[len(xtr),len(xva),len(xte)], dataset_shape=list(df.shape),
                   selection='First two accepted and first two rejected in the original seed-42 1000-row test sample among applicants whose raw-input and median-filled decisions agree at 0.5. No explanation metrics or true labels used for selection.',
                   preprocessing_class_flips_in_sample=int((~stable).sum()),
                   excluded_preprocessing_flip_ids=df.loc[raw.index[~stable], 'SK_ID_CURR'].astype(int).tolist(),
                   input_policy='Both methods explain identical training-median-filled inputs. Original code used raw NaN inputs for SHAP and filled inputs for LIME.',
                   lime_settings=dict(num_features=K,num_samples=NS,discretize_continuous=True,
                       kernel_width=float(.75*np.sqrt(len(names))), categorical_features=None,
                       figure_seed=42,repeat_seeds=list(range(42,52)),per_instance_pristine_explainer=True),
                   versions={p:importlib.metadata.version(p) for p in ['lightgbm','shap','lime','numpy','pandas','scikit-learn','matplotlib']},
                   python=sys.version, model_sha256=digest(model_path),
                   dataset_sha256=digest(ROOT/'application_train.csv'),
                   source_code_sha256=digest(ROOT/'csf_evaluation.py'),
                   category_mappings=mappings,training_medians=medians.to_dict())
    (OUT/'explanation_data.json').write_text(json.dumps(results,indent=2,allow_nan=False),encoding='utf-8')
    return results

INK, MUTED, RED, BLUE = '#17263c', '#526276', '#c44846', '#2475a8'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,
                     'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':INK,
                     'pdf.fonttype':42,'svg.fonttype':'none'})

def fmt(v):
    return f'{v:,.0f}' if float(v).is_integer() else f'{v:.3f}'

def plot_case(c, data, limits):
    fig = plt.figure(figsize=(16,9),facecolor='white')
    fig.add_artist(Rectangle((0,.87),1,.13,transform=fig.transFigure,color=INK,zorder=-1))
    fig.text(.035,.945,f"{c['decision'].upper()}  |  Applicant {c['applicant_id']}",fontsize=25,weight='bold',color='white')
    fig.text(.035,.899,'SHAP and LIME explain the same LightGBM input and the same default-class output',fontsize=13,color='#dae4f0')
    fig.text(.965,.943,f"Model score  {c['score']:.3f}",fontsize=20,weight='bold',color='white',ha='right')
    fig.text(.965,.899,'Accept < 0.500  |  Reject >= 0.500',fontsize=12,color='#dae4f0',ha='right')
    fig.text(.035,.821,'SHAP',fontsize=21,weight='bold')
    fig.text(.535,.821,'LIME',fontsize=21,weight='bold')
    fig.text(.035,.783,'Feature value + exact TreeSHAP attribution',fontsize=12,color=MUTED)
    fig.text(.535,.783,'Discretized condition + local surrogate weight',fontsize=12,color=MUTED)
    fig.text(.035,.749,f"Top-10 attribution coverage: {c['shap_top10_coverage']:.1%}   |   Raw-input score: {c['original_raw_score']:.3f}",fontsize=11,color=MUTED)
    r = c['lime_runs'][0]
    fig.text(.535,.749,f"Local fidelity R²: {r['r2']:.3f}   |   1,000 perturbations   |   seed 42",fontsize=11,color=MUTED)
    fig.add_artist(plt.Line2D([.505,.505],[.24,.83],transform=fig.transFigure,color='#dce4ec',lw=1))
    names = data['feature_names']
    top = c['shap_top_indices']
    labels = [f"{names[i]} = {fmt(c['input'][names[i]])}" + (' *' if names[i] in c['imputed_features'] else '') for i in top]
    labels.append(f"Other {len(names)-K} features (sum)")
    vals = [c['shap_values'][i] for i in top] + [c['shap_other_sum']]
    lime_labels = [t['condition'] for t in r['terms']]
    lime_vals = [t['weight'] for t in r['terms']]
    for side, ls, vs, xlabel in [(0,labels,vals,'SHAP value (model log-odds)'),
                                (1,lime_labels,lime_vals,'LIME weight (model-score units)')]:
        ax = fig.add_axes([.285 + .5*side,.325,.198,.377])
        ys = np.arange(len(vs))
        ax.barh(ys,vs,height=.60,color=[RED if v>0 else BLUE for v in vs],zorder=3)
        ax.set_yticks(ys,ls,fontsize=10)
        ax.tick_params(axis='y',length=0,pad=9)
        ax.set_ylim(10.7,-.7)
        lim=limits[side]
        ax.set_xlim(-lim,lim)
        ax.axvline(0,color='#7d8997',lw=.9)
        ax.grid(axis='x',color='#e8edf2',zorder=0)
        ax.set_xlabel(xlabel,fontsize=10,labelpad=9)
        ax.tick_params(axis='x',labelsize=9)
        for s in ax.spines.values(): s.set_visible(False)
        for yy,v in zip(ys,vs):
            ax.text(v+np.sign(v)*lim*.035,yy,f'{v:+.3f}',va='center',ha='left' if v>=0 else 'right',fontsize=9)
    fig.text(.035,.255,'Blue / negative: lowers default-class score',color=BLUE,fontsize=11)
    fig.text(.535,.255,'Red / positive: raises default-class score',color=RED,fontsize=11)
    for x in [.025,.525]:
        fig.add_artist(Rectangle((x,.098),.45,.128,transform=fig.transFigure,facecolor='#f1f5f9',edgecolor='none',zorder=-1))
    fig.text(.038,.195,'EXACT MODEL RECONSTRUCTION',fontsize=10,weight='bold')
    fig.text(.038,.165,f"Base {c['shap_base']:+.3f} + all contributions {sum(c['shap_values']):+.3f} = {c['shap_margin']:+.3f}",fontsize=11)
    fig.text(.038,.132,f"Logistic transform -> {c['score']:.3f} model score",fontsize=11)
    fig.text(.538,.195,'LOCAL APPROXIMATION CHECK',fontsize=10,weight='bold')
    fig.text(.538,.165,f"Surrogate {r['local_prediction']:.3f}  |  Model {c['score']:.3f}  |  Gap {abs(r['local_prediction']-c['score']):.3f}",fontsize=11)
    fig.text(.538,.132,'Condition boundaries describe bins; they are not approval thresholds.',fontsize=10)
    fig.text(.035,.065,'Same median-filled input for both methods; * = imputed value. Top 10 ranked within each method. Panel units differ.',fontsize=10,color=MUTED)
    fig.text(.035,.035,'Recomputed from paper code. Acceptance/rejection is a model rule, not a recorded lending decision. Score is uncalibrated.',fontsize=10,color=MUTED)
    return fig

def render(data):
    cases=data['cases']
    limits=[max(max(abs(v) for v in [c['shap_values'][i] for i in c['shap_top_indices']]+[c['shap_other_sum']]) for c in cases)*1.42,
            max(abs(t['weight']) for c in cases for t in c['lime_runs'][0]['terms'])*1.55]
    with PdfPages(OUT/'four_applicant_comparisons.pdf') as pdf:
        for c in cases:
            fig=plot_case(c,data,limits)
            stem=OUT/f"{c['case']}_{c['applicant_id']}_shap_lime"
            fig.savefig(str(stem)+'.png',dpi=300)
            fig.savefig(str(stem)+'.svg')
            fig.savefig(str(stem)+'.pdf')
            pdf.savefig(fig)
            plt.close(fig)
    rows=[]
    for c in cases:
        r=c['lime_runs'][0]
        rows.append({k:c[k] for k in ['case','applicant_id','source_row_index','csv_line','sample_position','decision','score','original_raw_score','observed_target','shap_top10_coverage','lime_10_seed_mean_pairwise_jaccard']} |
                    dict(lime_r2=r['r2'],lime_surrogate_score=r['local_prediction'],lime_absolute_score_error=abs(r['local_prediction']-c['score'])))
    pd.DataFrame(rows).to_csv(OUT/'case_summary.csv',index=False)
    all_contributions=[]
    for c in cases:
        for i,name in enumerate(data['feature_names']):
            all_contributions.append(dict(case=c['case'],applicant_id=c['applicant_id'],
                  feature=name,input_value=c['input'][name],
                  imputed=name in c['imputed_features'],shap_log_odds=c['shap_values'][i],
                  in_displayed_top10=i in c['shap_top_indices']))
    pd.DataFrame(all_contributions).to_csv(OUT/'full_shap_attributions.csv',index=False)
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':
    if '--render-only' in sys.argv:
        results=json.loads((OUT/'explanation_data.json').read_text(encoding='utf-8'))
    else:
        results=compute()
    render(results)
