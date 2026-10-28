# Reproduction instructions

## Environment

Python 3.11 is the recorded local runtime. `requirements-verify.txt` pins the dependencies used for artifact checking; `requirements.txt` adds the full experiment and plotting dependencies. These pins describe the available rerun environment, not the unknown original training environment. Relevant historical versions and data/model hashes are also embedded in the evidence JSON.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux alternative: source .venv/bin/activate
python -m pip install -r requirements.txt
```

On systems where LightGBM requires an OpenMP runtime, install the platform's OpenMP package. Headless plotting can use `MPLBACKEND=Agg`.

## Level 1: verify existing evidence (no full dataset)

```bash
python -m unittest discover -s tests -v
python c1_threshold_ablation.py
python c1_ablation_recheck.py
```

These commands do not retrain models. The tests read the 1,000-row saved input array, supplied rerun model, JSON artifacts, and PDFs. See `verification/README.md` for the local checks performed when packaging this repository.

## Level 2: redraw later evidence (no full dataset)

```bash
python paper_c3b_c6_evidence.py --render-only
python generate_applicant_comparisons.py --render-only
python generate_corrected_figures.py
```

These redraw saved results and can overwrite tracked figures. `generate_fig1_fig2_exact.py` and `generate_original_figs.py` instead plot embedded reported numbers; running them does not reproduce an experiment.

## Level 3: obtain data and rerun experiments

Download `application_train.csv` from the [official competition](https://www.kaggle.com/competitions/home-credit-default-risk/data) under its terms. Place it in the repository root. See [DATA.md](DATA.md) for the expected shape and SHA-256.

Use a **separate working clone** for reruns: the historical scripts write fixed paths and overwrite JSON/figures. Artifact checksum tests intentionally fail after such changes until outputs are reviewed and their manifest is deliberately refreshed.

### Original end-to-end evaluation

```bash
python -u csf_evaluation.py
```

This trains Logistic Regression, Random Forest, and LightGBM, computes explanations and CSF metrics, and writes checkpoints/figures. The script estimates 4–7 hours; hardware and package versions can change this substantially. Some checkpoint files from the original experiment were not available in the supplied workspace; the original mixed transcript is retained instead. Fixed seeds do not guarantee bitwise parity with a different training environment.

### Supplemental correction sequence

```bash
python -u rerun_supplemental.py
python -u rerun_analysis_C_fix.py
python recompute_downstream.py
python generate_corrected_figures.py
```

The second command is required: it replaces the initial fixed-seed consistency calculation with the corrected stochastic calculation. `recompute_downstream.py` intentionally retains paper-reported constants for criteria other than corrected C4; it is not a fully automatic end-to-end recomputation of all new metrics.

### Four-case analysis and full-sample evidence

The supplied rerun model and its metadata are already included, so the full-sample evidence script can use them directly once the original dataset is available:

```bash
python -u paper_c3b_c6_evidence.py
```

To retrain the four-case model and generate its outputs first:

```bash
python -u generate_applicant_comparisons.py
python finalize_applicant_figures.py
python -u paper_c3b_c6_evidence.py
```

Retraining can alter model hashes, early-stopping rounds, case selection, and downstream explanations. The full-sample script verifies dataset/model hashes before reuse. It supports local resume through a pickle checkpoint; no pickle checkpoint is distributed in this repository.

The existing figure-layout verifier expects a temporary directory:

```bash
python -c "from pathlib import Path; Path('tmp').mkdir(exist_ok=True)"
python verify_paper_c3b_c6_evidence.py
```

It recomputes aggregates, checks layout, and writes reports plus a presentation ZIP. The newer `tests/` suite provides read-only numerical checks suited to CI.

## Paper source

The selected PDF is the review artifact. `paper_corrected.tex` is preserved as supplied; compiling it needs a LaTeX installation with IEEEtran and its listed packages, using the root as the working directory. It has not been reconciled with the later corrected root figures. See [provenance](RESULTS_AND_PROVENANCE.md) before using a fresh compilation as a replacement manuscript.
