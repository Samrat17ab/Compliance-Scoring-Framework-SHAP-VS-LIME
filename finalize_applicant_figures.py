"""Verify and package the four generated applicant figures."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import lightgbm as lgb
import pymupdf
from PIL import Image

root = Path(__file__).resolve().parent
out = root/'output/applicant_comparisons'
data = json.loads((out/'explanation_data.json').read_text(encoding='utf-8'))
model = lgb.Booster(model_file=str(out/'retrained_lightgbm.txt'))
expected_pngs=[]
summary=['# The four selected cases', '',
         'All four decisions agree between raw and median-filled inputs. Both explanation panels use the filled input.', '',
         '| Decision | Applicant ID | Model score | Original raw score | LIME R² | LIME surrogate score |',
         '|---|---:|---:|---:|---:|---:|']
for c in data['cases']:
    x = pd.DataFrame([c['input']], columns=data['feature_names'])
    p=float(model.predict(x,num_threads=4)[0])
    assert abs(p-c['score']) < 1e-12
    assert (p>=.5) == (c['original_raw_score']>=.5)
    assert ('rejected' if p>=.5 else 'accepted') == c['decision']
    assert abs(c['shap_base']+sum(c['shap_values'])-c['shap_margin']) < 1e-7
    assert c['shap_repeat_max_abs_delta'] == 0
    assert c['lime_fixed_seed_repeat_max_abs_delta'] == 0
    assert len(c['lime_runs']) == 10
    for r in c['lime_runs']:
        assert abs(r['intercept']+sum(t['weight'] for t in r['terms'])-r['local_prediction']) < 1e-7
    stem=f"{c['case']}_{c['applicant_id']}_shap_lime"
    expected_pngs.append(stem+'.png')
    with Image.open(out/(stem+'.png')) as image:
        assert image.size == (4800,2700)
    r=c['lime_runs'][0]
    summary.append(f"| {c['decision'].capitalize()} | {c['applicant_id']} | {c['score']:.6f} | {c['original_raw_score']:.6f} | {r['r2']:.6f} | {r['local_prediction']:.6f} |")
summary.extend(['', '## Repeatability checks', '',
                'Exact same-configuration reruns: all SHAP values and all fixed-seed LIME coefficients reproduced exactly for each applicant.', '',
                '| Applicant ID | LIME top-10 mean pairwise Jaccard across seeds 42–51 |',
                '|---|---:|'])
for c in data['cases']:
    summary.append(f"| {c['applicant_id']} | {c['lime_10_seed_mean_pairwise_jaccard']:.6f} |")
summary.extend(['',f"Preprocessing flips the predicted class for {data['preprocessing_class_flips_in_sample']}/1,000 sampled applicants; these cases are excluded by the stated eligibility rule.", '',
                'All selections, values, and settings are preserved in explanation_data.json. See DEFENCE_NOTES.md for interpretation and limits.'])
(out/'CASE_RESULTS.md').write_text('\n'.join(summary)+'\n',encoding='utf-8')
pdf=pymupdf.open(out/'four_applicant_comparisons.pdf')
assert len(pdf)==4
for i,page in enumerate(pdf):
    assert str(data['cases'][i]['applicant_id']) in page.get_text()
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                assert x0>=0 and y0>=0 and x1<=page.rect.width and y1<=page.rect.height, span
    page.get_pixmap(matrix=pymupdf.Matrix(1.4,1.4)).save(root/f'tmp/pdfs/comparison_{i+1}.png')
# Remove only superseded figure files created by this task, verified under OUT.
for old in out.glob('*_shap_lime.*'):
    if old.suffix in {'.png','.pdf','.svg'} and old.with_suffix('.png').name not in expected_pngs:
        assert old.resolve().parent == out.resolve()
        old.unlink()
archive=root/'output/presentation_figures.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for name in expected_pngs+['four_applicant_comparisons.pdf','DEFENCE_NOTES.md','CASE_RESULTS.md','case_summary.csv']:
        z.write(out/name,arcname=name)
(out/'verification.json').write_text(json.dumps(dict(
    cases=4,accepted=2,rejected=2,png_dimensions=[4800,2700],pdf_pages=4,
    model_predictions_verified=True,raw_filled_decisions_agree=True,
    shap_additivity_verified=True,lime_surrogate_reconstruction_verified=True,
    same_seed_repeats_verified=True,pdf_text_inside_page_bounds=True),indent=2))
print('\n'.join(summary))
print(f'Archive: {archive}')
