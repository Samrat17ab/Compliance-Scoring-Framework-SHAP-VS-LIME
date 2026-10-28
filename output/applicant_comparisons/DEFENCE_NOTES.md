# Using the four applicant figures

Your conference defence focuses on **C3b (user comprehensibility)** and **C6 (human oversight support)**. Use the two corresponding sections below as your speaking notes. C7 is optional background only.

Each PNG is a 16:9, 4800 x 2700 image suitable for a presentation slide. Each applicant has SHAP on the left and LIME on the right. The PDF contains the same four figures, one per page. SVG and individual PDFs are also provided for vector-quality resizing.

These are newly computed examples from the original dataset and model configuration, not recovered outputs from the original paper run. They illustrate the explanations; they do not independently validate the paper's aggregate results or qualitative scores.

## What the figures mean

- The model predicts the default class. A score below 0.5 is labelled **accepted**; a score at or above 0.5 is labelled **rejected**, matching the threshold in the supplied code. These are simulated model decisions, not observed bank approvals or refusals. TARGET records repayment/default, not approval.
- Both panels explain the **same applicant, model, and median-filled input**. This deliberately resolves a mismatch in the original code, which explained raw missing-value inputs with SHAP but median-filled inputs with LIME. Original raw-input scores and missing fields are preserved in the accompanying data.
- Red contributions increase the default-class output; blue contributions decrease it. SHAP uses log-odds, while LIME weights use model-score units. **Do not compare bar lengths or magnitudes across methods.** Within each method, the axis scale is fixed across all four applicants.
- SHAP shows the ten largest absolute contributions plus the signed sum of the remaining 110 features. Its base value plus all contributions reconstructs the model's raw margin; the logistic transform gives the displayed score. Coverage uses absolute attribution mass, not predictive accuracy.
- LIME shows the ten conditions in its local linear explanation. Its R-squared measures fit to the model over the weighted perturbed neighbourhood. The displayed surrogate-to-model gap checks the fit at the applicant itself. Neither number is interchangeable with SHAP coverage.
- Model scores have not been calibrated. The training code uses class weighting, so a score of 0.7 should not be presented as a validated 70% real-world default probability.
- An asterisk on a SHAP value label indicates a training-median replacement for a missing input. LIME conditions refer to the same filled values. Keep raw field names and encoded categories in mind: a condition on an encoded category is not an ordered real-world threshold.

## C3b: user comprehensibility — SHAP 2.0/5, LIME 4.0/5

**Defensible wording:** “We assigned LIME a higher score because its conditions explicitly state the feature ranges associated with each contribution. SHAP's raw attribution requires interpreting a baseline and log-odds. These figures illustrate that difference in representation.”

Ask the judges to compare the actual feature/value label in the left panel with a range condition in the right panel. Use the same applicant so the comparison is concrete.

**Limit:** This is the authors' structured judgment, not a measured comprehension effect. The figures do not establish that readers understand LIME twice as well, nor do they uniquely justify the exact 2.0 and 4.0 ratings. Both methods retain technical feature names. A user study with comprehension questions, independent raters, and a prespecified rubric is needed to validate the scores. Display design can improve either method; the new presentation layout itself was not evaluated in the paper.

## C6: human oversight — SHAP 4.0/5, LIME 3.5/5

**Defensible wording:** “Both methods provide direction and magnitude. TreeSHAP additionally provides a full additive accounting of this model output. LIME provides a sparse approximation, so we inspect its local fidelity and prediction gap before relying on it for review.”

Use the SHAP reconstruction box and LIME approximation box to demonstrate what an overseer can check. Consult the full attribution file when a feature is omitted from the top ten.

**Limit:** The paper lists direction, magnitude, actionable feature-level information, and traceability, but does not specify a reproducible numeric conversion from four checklist items to 4.0 or three items to 3.5. Present these as provisional author-assigned ratings. Neither SHAP nor LIME proves causality or guarantees an intervention that changes the decision. A LIME bin boundary is not a loan approval threshold; a SHAP attribution is not a recommended change amount. The paper's claim that discretization limits actionability is an interpretation, not a demonstrated recourse experiment. Do not claim these figures prove that exact half-point difference.

## Optional background: C7 auditability — SHAP 5.0/5, LIME 2.0/5

Use this section only if the judges ask about repeatability. The generation script repeats TreeSHAP, repeats LIME with the same seed, and runs LIME with ten seeds (42–51). Each case's mean pairwise top-10 Jaccard is in `case_summary.csv`; full runs are in `explanation_data.json`.

**Defensible wording:** “For this fixed model and explainer configuration, TreeSHAP reproduced exactly. LIME also reproduced when its random state was reset, but its selected features varied across seeds. Our rating reflects the extra sampling-state management in this configuration.”

**Limit:** A fixed-seed LIME workflow is reproducible; a score of 2.0 is not evidence that LIME cannot be audited. Both methods require versioned preprocessing, inputs, model, explainer settings, and software environment; SHAP can also require its reference/background data. Input and model weights alone are not a complete audit record. These four-case reruns do not reproduce the paper's 100-instance consistency estimate.

## Provenance and reproduction

- The dataset and `csf_evaluation.py` already existed in the workspace. The attached text contains notebook commands and old console output as well as source code, so it was not executed wholesale.
- Only the selected LightGBM model was retrained. Logistic regression, random forest, and the full 1,000-case scoring study were not rerun because they are unnecessary for the four requested explanations.
- The stratified splits, seed 42, model hyperparameters, validation early stopping, and 1,000-row sampled test subset follow the original code. Training uses raw missing values as in that code; the input shown to both explainers is median-filled. Thread count is limited to four.
- Selection uses the first two accepted and first two rejected cases in the sampled test order **whose raw-input and median-filled classifications agree**. Neither actual target labels nor SHAP/LIME quality was used to select them. This eligibility condition keeps the labels consistent with the original raw-input pipeline; it is not a representative sample of all applicants. Candidate 116731 was excluded because its score changed from 0.544690 (raw; rejected) to 0.490649 (filled; accepted). The complete list of excluded IDs and the number of preprocessing-induced class flips in the 1,000-row sample are recorded in the JSON.
- LIME uses the original defaults: ten features, 1,000 perturbations, quartile discretization, and kernel width 0.75 times the square root of 120. No categorical-feature indices were passed, matching the source. Each displayed case starts from the same pristine seed-42 explainer state; it does not attempt to recreate its position in the original sequential RNG stream.
- The original saved model was absent. The new model stops at 190 rounds, compared with 204 in the uploaded console output. Its raw-test AUC is 0.759648, compared with 0.759 reported in the paper. These are similar performance results, not an assertion of identical fitted models or explanations.
- `explanation_data.json` records the exact inputs, all SHAP contributions, LIME conditions/weights and repeats, category mappings, training medians, versions, configuration, and model/dataset hashes. `retrained_lightgbm.txt` is the fitted model used for the figures.
- Run `.\.venv\Scripts\python.exe generate_applicant_comparisons.py` from the project root. The saved model from this run is reused if present. Use `--render-only` to redraw from the saved explanations without recomputing. If changing data or model settings, use a new output directory or explicitly remove the cached model after backing it up.

## Sources

- Supplied paper: Sections IV-C, IV-F, IV-G and V-C; Table V. The paper identifies C3b, C6, and C7 as qualitative and calls for independent evaluator studies. Your selected defence topics are C3b and C6; C7 is retained only as optional background.
- [SHAP TreeExplainer documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html): output units, additive reconstruction, and background/configuration requirements.
- [LIME documentation](https://lime-ml.readthedocs.io/en/latest/lime.html): discretization, random state, local linear explanations, and weighted R-squared.
- [LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#scale_pos_weight): class weighting and limitations of individual probability estimates.

These notes discuss the experimental evidence and the authors' scoring rationale. They do not validate the paper's statutory interpretations.
