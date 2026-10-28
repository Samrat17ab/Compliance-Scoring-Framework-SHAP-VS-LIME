"""Independently recompute requested summaries from saved per-instance data."""
from pathlib import Path
import json
import zipfile
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/paper_c3b_c6_evidence'
d=json.loads((OUT/'evidence.json').read_text())
p=d['per_instance']
assert len(p)==1000 and len({r['applicant_id'] for r in p})==1000
assert [r['applicant_id'] for r in p]==d['sample_applicant_ids']
assert [r['sample_position'] for r in p]==list(range(1000))
a=d['part_a']; b=d['part_b']; c=d['part_c']; t=a['traceability']
sv=np.load(OUT/'full_shap_and_inputs.npz')
assert sv['shap_values'].shape==(1000,120)
assert sv['inputs'].shape==(1000,120)
errors=np.abs(np.array([r['shap_base'] for r in p])+sv['shap_values'].sum(axis=1)-np.array([r['model_log_odds'] for r in p]))
assert np.isclose(errors.max(),t['shap_reconstruction_error_max'],rtol=0,atol=1e-14)
gaps=np.array([abs(r['lime']['surrogate_prediction']-r['score']) for r in p])
for stat,actual in [('mean',gaps.mean()),('median',np.median(gaps)),('p90',np.percentile(gaps,90))]:
    assert actual==t['lime_absolute_surrogate_gap_'+stat]
for method in ['shap','lime']:
    totals={k:0 for k in b['top10_composition'][method]}
    burden=gender=0
    for r in p:
        terms=r['shap_top10'] if method=='shap' else r['lime']['top10']
        assert len(terms)==10 and len({x['feature'] for x in terms})==10
        for x in terms:
            assert x['category']==b['feature_category_map'][x['feature']]
            totals[x['category']]+=1
        burden+=sum(x['category'] in ['document_flags','encoded_categoricals'] for x in terms)>=5
        gender+=any(x['feature']=='CODE_GENDER' for x in terms)
    assert sum(totals.values())==10000
    for cat,count in totals.items():
        assert count==b['top10_composition'][method][cat]['total_items']
        assert np.isclose(count/100,b['top10_composition'][method][cat]['mean_pct_of_top10'])
    assert burden==b['at_least_5_document_flags_or_encoded_categoricals'][method]['instance_count']
    assert gender==b['code_gender_in_top10'][method]['instance_count']
bin_count=sum('<' in term['condition'] or '>' in term['condition'] for r in p for term in r['lime']['top10'])
assert bin_count==a['representation']['lime_bin_or_range_item_count']
assert bin_count/100==a['representation']['lime_bin_or_range_item_pct']
for decision in ['accepted','rejected']:
    group=[r for r in p if (r['score']<.5)==(decision=='accepted')]
    median=float(np.median([r['lime']['r2'] for r in group]))
    chosen=min(group,key=lambda r:(abs(r['lime']['r2']-median),r['sample_position']))
    selected=c['selected'][decision]
    assert chosen['applicant_id']==selected['applicant_id']
    assert selected['group_count']==len(group)
    assert median==selected['group_median_lime_r2']
for filename in ['accepted_example.png','rejected_example.png','summary_1000.png']:
    with Image.open(OUT/filename) as im:
        assert im.size==(3200,1800)
        im.resize((1600,900)).save(ROOT/'tmp'/filename)

# Check the final vector layout before inspecting raster previews.
import paper_c3b_c6_evidence as plotting
for decision in ['accepted','rejected']:
    fig=plotting.example_slide(d,decision)
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax in fig.axes:
        labels=ax.get_yticklabels()
        assert all(label.get_fontsize()>=14 for label in labels)
        boxes=sorted([label.get_window_extent(renderer) for label in labels],key=lambda box:box.y0)
        assert all(box.x0>=0 and box.x1<=fig.bbox.width for box in boxes)
        assert all(b.y0>=a.y1 for a,b in zip(boxes,boxes[1:])), (decision,'Feature labels overlap')
    plotting.plt.close(fig)

lines=['# Full-sample results: C3b and C6', '',
       'N = 1,000 applicants; 10 features per method per applicant; 10,000 explanation items per method.', '',
       'Paper explainer settings with shared median-filled inputs. Newly computed with the existing retrained paper-settings model; no sensitivity variants.', '',
       '## Part A — C6 proxies', '',
       f"- SHAP maximum absolute reconstruction error: **{t['shap_reconstruction_error_max']:.12g} log-odds**.",
       f"- LIME absolute surrogate gap, mean: **{t['lime_absolute_surrogate_gap_mean']:.9f}**; median: **{t['lime_absolute_surrogate_gap_median']:.9f}**; 90th percentile: **{t['lime_absolute_surrogate_gap_p90']:.9f}**, all in model probability-scale units.",
       f"- LIME bin/range items: **{bin_count}/10,000 ({bin_count/100:.2f}%)**; SHAP bin/range labels: **0%**, exact model-input value labels: **100%**.",
       '- These are representation and numerical-accounting proxies. Neither method proves causal actionability. SHAP and LIME error scales differ.', '',
       '## Part B — C3b content proxies', '',
       '| Category | SHAP mean % of top ten | LIME mean % of top ten | SHAP item count | LIME item count |',
       '|---|---:|---:|---:|---:|']
for cat in b['top10_composition']['shap']:
    s=b['top10_composition']['shap'][cat]; l=b['top10_composition']['lime'][cat]
    lines.append(f"| {cat} | {s['mean_pct_of_top10']:.2f}% | {l['mean_pct_of_top10']:.2f}% | {s['total_items']} | {l['total_items']} |")
lines.extend(['','| Instance measure | SHAP | LIME |','|---|---:|---:|'])
for key,label in [('at_least_5_document_flags_or_encoded_categoricals','At least 5 document flags / encoded categoricals'),('code_gender_in_top10','CODE_GENDER in top ten')]:
    lines.append(f"| {label} | {b[key]['shap']['pct_instances']:.2f}% ({b[key]['shap']['instance_count']}/1,000) | {b[key]['lime']['pct_instances']:.2f}% ({b[key]['lime']['instance_count']}/1,000) |")
lines.extend(['',f"Label-encoded columns ({len(b['label_encoded_columns'])}):",''])
lines.extend('- `'+col+'`' for col in b['label_encoded_columns'])
lines.extend(['','## Part C — Selected examples','',c['rule'],'',
              '| Class | ID | Group n | Score | Class median R² | Selected R² | Distance |',
              '|---|---:|---:|---:|---:|---:|---:|'])
for label,s in c['selected'].items():
    lines.append(f"| {label} | {s['applicant_id']} | {s['group_count']} | {s['score']:.9f} | {s['group_median_lime_r2']:.9f} | {s['selected_lime_r2']:.9f} | {s['absolute_distance_to_group_median']:.9g} |")
lines.extend(['','See README.md for category definitions, model provenance, configuration, and interpretation limits. The JSON preserves unrounded numbers.'])
(OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'verification.json').write_text(json.dumps(dict(n=1000,items_per_method=10000,all_aggregates_recomputed=True,
                  sample_ids_unique=True,median_selection_verified=True,png_dimensions=[3200,1800]),indent=2))
with zipfile.ZipFile(ROOT/'output/paper_c3b_c6_presentation_pack.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['accepted_example.png','rejected_example.png','summary_1000.png','evidence.json','README.md','RESULTS.md','verification.json']:
        z.write(OUT/name,arcname=name)
    z.write(ROOT/'paper_c3b_c6_evidence.py',arcname='source/paper_c3b_c6_evidence.py')
print('\n'.join(lines))
