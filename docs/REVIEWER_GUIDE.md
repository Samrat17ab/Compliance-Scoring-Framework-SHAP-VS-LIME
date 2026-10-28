# Reviewer guide

## A short reading route

1. Read the [selected manuscript](../paper/Evaluating_SHAP_and_LIME_Against_EU_AI_Act_A.pdf).
2. Read [results and provenance](RESULTS_AND_PROVENANCE.md) to distinguish reported results, supplemental corrections, and later shared-input reruns.
3. Inspect the [full-sample results](../output/paper_c3b_c6_evidence/RESULTS.md) and their [configuration](../output/paper_c3b_c6_evidence/README.md).
4. Run the [artifact tests](REPRODUCIBILITY.md). No full training-table download is needed.

## Claim-to-evidence map

| Question | Code | Saved evidence |
|---|---|---|
| How were models trained and CSF components calculated? | [csf_evaluation.py](../csf_evaluation.py) | [Original transcript](../archive/home%20credit%20code%20v1%20paper.txt), selected paper |
| What happens after putting C4 SHAP in probability space? | [rerun_supplemental.py](../rerun_supplemental.py), [recompute_downstream.py](../recompute_downstream.py) | [A: C4](../supplemental_A_c4_probability.json), [D: scores](../supplemental_D_corrected_scores.json) |
| How sensitive is LIME to its configuration? | [rerun_supplemental.py](../rerun_supplemental.py) | [B: sensitivity](../supplemental_B_lime_sensitivity.json) |
| Is stochastic consistency different from fixed-seed repeatability? | [rerun_analysis_C_fix.py](../rerun_analysis_C_fix.py) | [C: paired Jaccard](../supplemental_C_paired_jaccard.json), [corrected log](../archive/supplemental_C_fix.log) |
| How dependent is C1 ranking on thresholds? | [c1_threshold_ablation.py](../c1_threshold_ablation.py), [c1_ablation_recheck.py](../c1_ablation_recheck.py) | Run transcripts in [verification](../verification/) |
| What do paired explanations look like? | [generate_applicant_comparisons.py](../generate_applicant_comparisons.py) | [Case summary](../output/applicant_comparisons/CASE_RESULTS.md), [defence notes](../output/applicant_comparisons/DEFENCE_NOTES.md) |
| What evidence supports the C3b/C6 discussion? | [paper_c3b_c6_evidence.py](../paper_c3b_c6_evidence.py) | [Per-instance JSON](../output/paper_c3b_c6_evidence/evidence.json), [results](../output/paper_c3b_c6_evidence/RESULTS.md) |
| Can the saved aggregates be independently checked? | [tests/test_artifacts.py](../tests/test_artifacts.py), [existing verifier](../verify_paper_c3b_c6_evidence.py) | [Local verification](../verification/README.md), optional CI template |

## Figure provenance

| Files / generator | Meaning |
|---|---|
| `generate_fig1_fig2_exact.py` | Redraws hard-coded paper-reported values; **not** a new experiment |
| `generate_fig1_fig2.py` | Retrains and recomputes kernel-sweep and consistency results |
| `generate_original_figs.py` | Redraws original reported figures from embedded values |
| `generate_corrected_figures.py` | Draws C4/radar/sensitivity figures from supplemental JSON and some paper-reported constants |
| Root `fig1_*`, `fig2_*` | Existing saved figures; corresponding reported-value rendering code is included |
| Root `fig3_*`, `fig4_*`, `fig5_*`, `fig_weight_*` | Existing saved supplemental/corrected figures; distinguish them from figures embedded in the selected PDF |
| `output/paper_c3b_c6_evidence/*.png` | Newer shared-input analysis, documented alongside its outputs |

The selected manuscript PDF is preserved unchanged. Root figure files and the LaTeX source are independently versioned artifacts, not a guarantee that recompilation recreates that PDF.

## Useful limitations to examine

- C1 uses different measurement constructs for the two methods.
- C3b, C6, and C7 include author judgments rather than user-study validation.
- C5 is a model property shared by both explanation methods.
- Top-ten visibility is different from an attribution being computed at all.
- Input imputation and explainer random-number state affect the comparison.
- A single dataset/model/configuration cannot establish universal superiority.
- Country and nationality are not directly identified in the application table.
