# C3b and C6 evidence: full 1,000-instance sample

The three PNGs are 3200 x 1800 pixels, designed at 13 1/3 x 7.5 inches (the exact 16:9 size conventionally called 13.33 x 7.5), 240 dpi. Example feature labels are at least 14 pt at this slide size.

## Files

- `accepted_example.png` and `rejected_example.png`: applicants selected by distance to their predicted class's median LIME R²; eight features displayed per method.
- `summary_1000.png`: full-sample statistics, computed with ten features per explanation.
- `evidence.json`: every result for Parts A–C, counts, denominators, per-instance explanations, sample IDs, categorical encodings, feature category definitions, settings, versions and hashes.
- `full_shap_and_inputs.npz`: all 120 SHAP contributions and exact median-filled inputs for all 1,000 applicants, with names and IDs.
- `lightgbm_model.txt`: the model used for this run.
- `RESULTS.md`: numerical results and selected IDs, generated after completion.

## Configuration and scope

This is a **paper-explainer-configuration, shared-input rerun**, not a hyperparameter sensitivity variant. There are no sensitivity variants in this output. Both methods receive identical training-median-filled rows, as requested; the original paper code gave TreeSHAP raw missing values and LIME filled inputs. Thus these results should be cited as a new aligned-input analysis, not as numbers recovered from the published experiment.

The existing retrained LightGBM from the earlier task is reused. It uses the original paper's training settings; it has 190 boosting rounds and raw-test AUC 0.759648, compared with 204 rounds and reported AUC 0.759 in the uploaded console output. The original fitted paper model was unavailable. The data splits and 1,000-row sample are reproduced using random_state=42.

TreeSHAP is called as `shap.TreeExplainer(model)` with defaults. LIME uses `discretize_continuous=True`, `num_features=10`, `num_samples=1000`, with **no categorical_features argument**. A single explainer is initialized with random_state=42 and used sequentially in sample order, exactly as in the paper's main loop. There is no per-applicant seed reset. CPU threading is limited for execution efficiency; no explanation hyperparameters are tuned.

## Part A: interpretation

SHAP reconstruction uses the base value and **all 120 contributions**, compared with the model's raw log-odds margin. LIME's gap is the absolute difference between its ten-feature surrogate prediction and the model's probability-scale output at the same applicant. These use different scales and test different properties: do not compare their numerical magnitudes as a ratio. LIME's neighbourhood R² is separately recorded and used only for example selection.

“Bins/ranges” counts LIME condition labels containing `<` or `>`. The actual model input is attached to SHAP labels; this is a presentation choice using the input, not an additional output produced by the SHAP attribution algorithm. Missing numerical values are training medians, not observed applicant facts, and are marked with an asterisk.

This measures **value-versus-bin representation**, not validated actionability. A bin on a binary flag may identify a single value exactly. Neither method establishes a causal change, feasible recourse, or an approval threshold. LIME's surrogate gap and SHAP's additive identity alone cannot establish full operational auditability or oversight quality.

## Part B: prespecified categories

Each feature belongs to exactly one category; this prevents double-counting:

1. Document flags: `FLAG_DOCUMENT_*`.
2. Encoded categoricals: every original object/string column transformed by LabelEncoder. The complete list and mappings are in the JSON and results file. This includes CODE_GENDER and string-valued ownership flags.
3. Opaque scores: `EXT_SOURCE_*`.
4. Plain-meaning fields: `AMT_*`, `CNT_*`, names containing `_CNT_`, and `DAYS_BIRTH`, `DAYS_EMPLOYED`, `OWN_CAR_AGE`. This is an explicit amounts/counts/age/employment definition; it is not a claim that every reader understands these names, negative day offsets, or sentinel codes.
5. Other: everything else, including other time offsets, regional ratings, housing aggregates, and remaining flags. These are kept separate rather than automatically labelled plain-meaning.

For each method, the mean percentage of top-ten items is the mean of 1,000 per-instance percentages. Because every list has ten items, this equals the percentage of its 10,000 total items. The “at least five” rate uses the union of document flags and encoded categoricals, with 1,000 instances as denominator. Gender appearance is also an instance percentage and measures **top-ten appearance**, not whether SHAP computed a gender contribution at all.

These are content proxies for C3b, not measured human comprehension. They do not independently validate an exact author-assigned score or guarantee LIME is more readable than SHAP.

## Part C: automatic selection

Classify all 1,000 applicants using the shared filled-input model score: accepted below 0.5, rejected at or above 0.5. Within each group, compute the median LIME R² and choose the instance with the smallest absolute distance from that median. Ties are resolved by earliest position in the original sampled order. There is **no additional exclusion**, including no exclusion for disagreement between raw-input and filled-input predictions. These are model decisions, not recorded lending outcomes.

## Reproduction

From the workspace root, run:

```powershell
& '.\.venv\Scripts\python.exe' -u paper_c3b_c6_evidence.py
```

Use `--render-only` to redraw the PNGs from the saved JSON. A local checkpoint preserves the sequential LIME random-number state; it is not a per-instance reseeding scheme. The checkpoint is an internal artifact and should not be loaded from an untrusted source. The model and data hashes are checked before reuse.
