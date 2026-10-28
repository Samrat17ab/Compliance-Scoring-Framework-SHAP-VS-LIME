# Verification record

Repository preparation date: **28 October 2026** (Asia/Calcutta).

The following checks were executed locally with Python 3.11 using the existing experiment environment:

| Check | Result | Record |
|---|---|---|
| Independent artifact suite | 12 tests passed | [unittest.txt](unittest.txt) |
| Original C1 threshold ablation | Exit 0 | [c1_threshold_ablation.txt](c1_threshold_ablation.txt) |
| Original independent C1 recheck | Exit 0 | [c1_ablation_recheck.txt](c1_ablation_recheck.txt) |
| Existing full-sample aggregate and figure-layout verifier | Exit 0 | [existing_evidence_verifier.txt](existing_evidence_verifier.txt) |
| Selected manuscript | Six pages rendered and visually inspected | Unchanged source PDF in `paper/` |
| Main documentation links | Local targets checked | README and `docs/` |

The independent suite checks the model predictions on all 1,000 saved inputs, SHAP reconstruction and top-ten arrays, LIME gaps, content counts, median-based example selection, paired Wilcoxon result, corrected scores, weighting simulation, manuscript text, image dimensions, and content hashes. This is saved-artifact verification; the original multi-hour training experiment was not rerun during repository preparation.

`artifact_manifest.json` records SHA-256 and byte size for shipped numerical and visual artifacts. `source_manifest.json` records the original root Python scripts using normalized text line endings, so Git's platform line-ending conversion does not change source-preservation checks.

Historical logs live in `archive/`; they are distinct from these newly executed checks. Passing a script's existing assertions does not validate every sentence it prints. In particular, the C1 recheck's R²-only loop starts at -95%, so its printed “first” -95% result is the first scanned satisfying value, **not** the minimum reduction needed for a ranking change. The code is preserved unchanged for provenance.

To repeat the numerical checks:

```bash
python -m pip install -r requirements-verify.txt
python -m unittest discover -s tests -v
```

Hosted GitHub Actions checks are **not enabled** in this initial upload: the publishing connection can create repositories and push code but lacks the `workflow` scope. The ready-to-use workflow is [tools/github-actions-verify.yml](../tools/github-actions-verify.yml). A repository maintainer with workflow-write access can copy it to `.github/workflows/verify.yml` to enable the same suite. The badge in the README describes local verification, not hosted CI.
