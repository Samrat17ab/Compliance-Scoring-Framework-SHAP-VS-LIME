# The four selected cases

All four decisions agree between raw and median-filled inputs. Both explanation panels use the filled input.

| Decision | Applicant ID | Model score | Original raw score | LIME R² | LIME surrogate score |
|---|---:|---:|---:|---:|---:|
| Accepted | 209765 | 0.273784 | 0.454203 | 0.060965 | 0.397238 |
| Accepted | 342011 | 0.088841 | 0.141174 | 0.292716 | 0.198604 |
| Rejected | 218332 | 0.797803 | 0.814378 | 0.549487 | 0.525564 |
| Rejected | 425755 | 0.834881 | 0.821325 | 0.489362 | 0.582780 |

## Repeatability checks

Exact same-configuration reruns: all SHAP values and all fixed-seed LIME coefficients reproduced exactly for each applicant.

| Applicant ID | LIME top-10 mean pairwise Jaccard across seeds 42–51 |
|---|---:|
| 209765 | 0.344766 |
| 342011 | 0.354831 |
| 218332 | 0.426150 |
| 425755 | 0.385287 |

Preprocessing flips the predicted class for 149/1,000 sampled applicants; these cases are excluded by the stated eligibility rule.

All selections, values, and settings are preserved in explanation_data.json. See DEFENCE_NOTES.md for interpretation and limits.
