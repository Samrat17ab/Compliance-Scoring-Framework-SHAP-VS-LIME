# Results and provenance

## 1. Selected manuscript and original evidence

The main paper is `paper/Evaluating_SHAP_and_LIME_Against_EU_AI_Act_A.pdf`, selected by the author for this repository. It reports overall CSF scores of 4.12 (SHAP) and 2.50 (LIME), stochastic LIME Jaccard 0.381, and SHAP Jaccard 1.000.

`archive/home credit code v1 paper.txt` is the original supplied mixed code-and-console-output transcript, preserved byte-for-byte in its original Windows-1252 encoding. It is historical evidence, not an executable Python file. `csf_evaluation.py` is the existing executable script. All pre-existing Python scripts were copied without changing scientific calculations.

The earlier PDF under `paper/archive/` is kept for version history. It is not the main manuscript. `paper_corrected.tex` is the existing local source, not a newly reconciled camera-ready source. Its original figure references resolve to the root figures, some of which were subsequently corrected. Compiling it therefore does **not** promise byte-for-byte reproduction of the selected PDF or agreement between every caption and figure. Read the selected PDF for the supplied manuscript; consult the supplemental JSON for corrected numerical results.

## 2. Supplemental corrections

### C4: attribution output scales

The original comparison placed SHAP log-odds contributions alongside LIME probability-scale coefficients. Their numerical ratio is not a like-for-like effect-size comparison. `rerun_supplemental.py` computes an interventional probability-space SHAP analysis; `supplemental_A_c4_probability.json` records a gap of approximately 0.0389505. This also changes the SHAP explanation configuration, not merely the displayed units.

`recompute_downstream.py` uses the rounded corrected value 0.038951 and paper-reported inputs for the other criteria. It produces SHAP 4.00 and LIME 2.50 in `supplemental_D_corrected_scores.json`. This is a partial downstream correction, not a complete rerun of all original metrics. The stored 4.915 ratio uses a paper-reported LIME gap of 0.007925; it is not a causal effect or evidence of legal compliance.

The corrected Monte Carlo analysis ranks SHAP higher in 2,963 of 3,000 sampled weight vectors (98.77%). The transcript's original figure is 2,971 of 3,000 (99.03%). The threshold perturbation output averages five quantitative components, unlike the eight-component overall CSF. Do not compare those aggregates as identical scoring schemes.

### C2: stochastic versus fixed-seed LIME

The initial `rerun_supplemental.py` consistency section reset the explainer's seed per run and reproduced fixed-seed explanations. Its historical log reports LIME Jaccard 1.0 and an undefined Wilcoxon p-value. That is **not** the corrected stochastic result.

`rerun_analysis_C_fix.py` reuses a single explainer across runs and overwrites `supplemental_C_paired_jaccard.json` with the corrected evidence: SHAP mean Jaccard 1.0, LIME approximately 0.373402, W = 0, p ≈ 3.89634e-18. `archive/supplemental_C_fix.log` records this correction. Run the correction after the general supplemental script when reproducing.

### LIME configuration matters

`supplemental_B_lime_sensitivity.json` records experiments on 150 applicants per configuration. Disabling discretization increases mean R² from approximately 0.248 to 0.521 in that experiment. Consequently, low fidelity should not be described as an intrinsic property of all LIME configurations.

## 3. Later paired applicant analyses

`output/applicant_comparisons/` uses a retrained LightGBM with the paper's settings. Its raw-test AUC is approximately 0.759648 and it has 190 boosting rounds, compared with 204 rounds in the supplied original console output. The original fitted model was unavailable.

Both explainers receive training-median-filled inputs. Original paper code passed raw missing values to TreeSHAP while using filled inputs for LIME. This input alignment can change model scores: 149 of the 1,000 sampled applicants change predicted class after filling. The four-case selection excludes those applicants; its documented selection policy therefore differs from the full-sample analysis below.

## 4. Full-sample C3b/C6 evidence

`output/paper_c3b_c6_evidence/` evaluates all 1,000 sampled applicants, ten features per explanation, using identical median-filled inputs and the retrained model. LIME uses a single seed-42 explainer sequentially, with discretization enabled, 1,000 perturbations, and no `categorical_features` argument.

Key saved results:

| Measurement | Value |
|---|---:|
| SHAP maximum additive reconstruction error | 7.9936e-15 log-odds |
| LIME mean absolute surrogate gap | 0.0878796 probability-scale units |
| Gender in SHAP top ten | 809 / 1,000 |
| Gender in LIME top ten | 97 / 1,000 |
| At least five document flags / encoded categoricals, SHAP | 2 / 1,000 |
| At least five document flags / encoded categoricals, LIME | 870 / 1,000 |

SHAP reconstruction uses **all 120 contributions**, not just the displayed top ten. The two error quantities use different units and answer different questions. Content categories and value/bin displays are proxies, not human-subject comprehension measurements, causal recourse, or validation of particular checklist scores.

Examples are selected nearest to the median LIME R² within each predicted class, without excluding preprocessing flips. The selected IDs are 140663 and 296065. “Accepted/rejected” is a model-threshold label at 0.5, not an observed bank decision. Class-weighted model outputs have not been established as calibrated risk probabilities.

## Verification boundary

The repository tests independently recalculate stored aggregates, compare predictions from the supplied rerun model with saved inputs, and check checksums. They do not establish correctness of the paper's regulatory interpretation, originality claims, causal conclusions, or externally valid fairness. They do not recreate the missing original fitted model. Existing logs are historical artifacts; the new `verification/` report records checks actually executed during repository preparation.
