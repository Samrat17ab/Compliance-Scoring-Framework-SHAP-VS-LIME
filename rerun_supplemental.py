#!/usr/bin/env python3
"""
Supplemental rerun — targeted fixes for reviewer points 4, 5, 6.
==================================================================
Reuses the EXACT preprocessing / split / LightGBM training / sampling
logic from csf_evaluation.py so the model and the 1000-instance sample
are identical to the published run. Does NOT touch C1, C2(original),
C3a, C3b, C5, C6, C7 or the LR/RF comparison — those stay as published.

Adds three new analyses:
  A) C4 bias gap recomputed in PROBABILITY space (interventional SHAP)
     to fix the log-odds vs. probability unit mismatch vs. LIME.
  B) LIME parameter sensitivity: perturbation sample size, num_features
     (K), and discretization on/off.
  C) Raw per-instance Jaccard arrays for SHAP & LIME consistency runs,
     aggregated per-instance for a proper PAIRED test (Wilcoxon),
     instead of treating 4,500 pseudo-replicated pairs as independent.

SMOKE_TEST=1 env var runs a tiny fast version to validate the code path
before committing to the multi-hour full run.
"""
import json
import os
import time
import warnings
from itertools import combinations

import lightgbm as lgb
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
np.random.seed(42)

SMOKE = os.environ.get('SMOKE_TEST', '0') == '1'

SAMPLE_N     = 1000
N_CONSIST    = 100
N_RUNS       = 10
K            = 10
LIME_SAMPLES = 1000
RANDOM_STATE = 42
GENDER_COL   = 'CODE_GENDER'

if SMOKE:
    SAMPLE_N     = 30
    N_CONSIST    = 6
    N_RUNS       = 3
    print("*** SMOKE TEST MODE ***")

t0 = time.time()
def elapsed():
    return f"[{time.time()-t0:6.1f}s]"


def jaccard(s1, s2):
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)


# -----------------------------------------------------------------------------
# STEP 1 — DATA (identical to csf_evaluation.py)
# -----------------------------------------------------------------------------
print(elapsed(), "Loading data...")
df = pd.read_csv('application_train.csv')
y = df['TARGET']
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
missing_frac = X.isnull().sum() / len(X)
cols_to_drop = missing_frac[missing_frac == 1.0].index.tolist()
X = X.drop(cols_to_drop, axis=1)
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_temp)

train_medians  = X_train.median(numeric_only=True)
X_train_filled = X_train.fillna(train_medians)
X_test_filled  = X_test.fillna(train_medians)
feature_names  = list(X_train.columns)
print(elapsed(), f"Train {X_train.shape} Val {X_val.shape} Test {X_test.shape}")

# -----------------------------------------------------------------------------
# STEP 2 — TRAIN LIGHTGBM ONLY (same config as csf_evaluation.py)
# -----------------------------------------------------------------------------
print(elapsed(), "Training LightGBM (identical config/seed to published run)...")
scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
lgb_train_ds = lgb.Dataset(X_train, label=y_train)
lgb_val_ds   = lgb.Dataset(X_val, label=y_val, reference=lgb_train_ds)
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9,
    'bagging_fraction': 0.8, 'bagging_freq': 5,
    'scale_pos_weight': scale_pw, 'verbose': -1, 'random_state': RANDOM_STATE,
}
model = lgb.train(
    lgb_params, lgb_train_ds, num_boost_round=1000,
    valid_sets=[lgb_val_ds],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)
from sklearn.metrics import roc_auc_score
lgb_prob = model.predict(X_test)
lgb_auc  = roc_auc_score(y_test, lgb_prob)
print(elapsed(), f"LightGBM AUC={lgb_auc:.4f} (published: 0.759) — sanity check")

# -----------------------------------------------------------------------------
# STEP 3 — SAME 1000-INSTANCE SAMPLE
# -----------------------------------------------------------------------------
X_sample_raw    = X_test.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
X_sample_filled = X_test_filled.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
X_consist_filled = X_sample_filled.iloc[:N_CONSIST].copy()
X_consist_raw     = X_sample_raw.iloc[:N_CONSIST].copy()
gender_idx = feature_names.index(GENDER_COL)

def predict_fn(x):
    d = pd.DataFrame(x, columns=feature_names)
    p = model.predict(d)
    return np.column_stack([1 - p, p])

results = {}

# =============================================================================
# ANALYSIS A — C4 bias gap in PROBABILITY space
# =============================================================================
print(elapsed(), "=" * 60)
print(elapsed(), "ANALYSIS A: C4 in probability space")
print(elapsed(), "=" * 60)

# A1) sanity check: reproduce the ORIGINAL log-odds default explainer
explainer_default = shap.TreeExplainer(model)
sv_default = explainer_default.shap_values(X_sample_raw)
gender_vals = X_sample_raw[GENDER_COL].values
valid_mask  = ~np.isnan(gender_vals.astype(float))
unique_g    = np.unique(gender_vals[valid_mask])
gender_shap_default = sv_default[:, gender_idx]
grp_default = {int(g): float(gender_shap_default[gender_vals == g].mean()) for g in unique_g}
sk = sorted(grp_default.keys())
gap_default = float(abs(grp_default[sk[0]] - grp_default[sk[1]]))
print(elapsed(), f"  Reproduced log-odds gap: {gap_default:.4f}  (published: 0.2696)  groups={grp_default}")

# A2) linearized probability approximation (cheap, uses default log-odds values)
preds_sample = model.predict(X_sample_raw)  # probabilities
local_deriv  = preds_sample * (1 - preds_sample)  # d(sigmoid)/d(logit)
gender_shap_prob_approx = gender_shap_default * local_deriv
grp_approx = {int(g): float(gender_shap_prob_approx[gender_vals == g].mean()) for g in unique_g}
gap_approx = float(abs(grp_approx[sk[0]] - grp_approx[sk[1]]))
print(elapsed(), f"  Linearized probability-space approx gap: {gap_approx:.6f}  groups={grp_approx}")

# A3) exact interventional probability-space SHAP (background = 50 filled train rows)
background = X_train_filled.sample(50, random_state=RANDOM_STATE)
print(elapsed(), "  Computing exact interventional probability-space SHAP "
                  f"(background=50, n={SAMPLE_N})... this is the slow part of Analysis A")
explainer_prob = shap.TreeExplainer(
    model, data=background, model_output="probability",
    feature_perturbation="interventional")
sv_prob = explainer_prob.shap_values(X_sample_filled)
gender_vals_filled = X_sample_filled[GENDER_COL].values
gender_shap_prob = sv_prob[:, gender_idx]
grp_prob = {int(g): float(gender_shap_prob[gender_vals_filled == g].mean())
            for g in np.unique(gender_vals_filled)}
skp = sorted(grp_prob.keys())
gap_prob_exact = float(abs(grp_prob[skp[0]] - grp_prob[skp[1]]))
print(elapsed(), f"  EXACT interventional probability-space gap: {gap_prob_exact:.6f}  groups={grp_prob}")

results['analysis_A_c4_probability_space'] = {
    'reproduced_log_odds_gap': gap_default,
    'reproduced_log_odds_groups': grp_default,
    'linearized_prob_approx_gap': gap_approx,
    'linearized_prob_approx_groups': grp_approx,
    'exact_interventional_prob_gap': gap_prob_exact,
    'exact_interventional_prob_groups': grp_prob,
    'lime_gap_for_reference': 0.0079,
}
with open('supplemental_A_c4_probability.json', 'w') as f:
    json.dump(results['analysis_A_c4_probability_space'], f, indent=2)
print(elapsed(), "  [CHECKPOINT] supplemental_A_c4_probability.json saved")

# =============================================================================
# ANALYSIS B — LIME parameter sensitivity (sample size, K, discretization)
# =============================================================================
print(elapsed(), "=" * 60)
print(elapsed(), "ANALYSIS B: LIME parameter sensitivity")
print(elapsed(), "=" * 60)

N_SENS = 8 if SMOKE else 150

def run_lime_config(num_samples, num_features, discretize, n, explainer=None):
    if explainer is None:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train_filled.values, feature_names=feature_names,
            class_names=['Repaid', 'Default'], mode='classification',
            discretize_continuous=discretize, random_state=RANDOM_STATE)
    r2s, gender_appear = [], 0
    complexities = []
    def struct_complexity(names):
        F = len(names)
        Nc = sum(1 for nm in names if any(op in nm for op in ['>', '<', '<=', '>=', '=']))
        Lb = float(np.mean([len(nm) for nm in names])) if names else 0.0
        return 0.4*(F/10) + 0.4*(Nc/10) + 0.2*(Lb/20)
    for i in range(n):
        try:
            exp = explainer.explain_instance(
                data_row=X_sample_filled.iloc[i].values, predict_fn=predict_fn,
                num_features=num_features, num_samples=num_samples)
            r2s.append(float(exp.score))
            names = [str(f[0]) for f in exp.as_list()]
            complexities.append(struct_complexity(names))
            if any(GENDER_COL.lower() in nm.lower() for nm in names):
                gender_appear += 1
        except Exception:
            pass
    return {
        'r2_mean': float(np.mean(r2s)) if r2s else None,
        'r2_std':  float(np.std(r2s)) if r2s else None,
        'complexity_mean': float(np.mean(complexities)) if complexities else None,
        'gender_appear_rate': gender_appear / n,
        'n_success': len(r2s),
    }

sens_results = {'sample_size': {}, 'num_features': {}, 'discretization': {}}

sample_sizes = [250, 500, 1000, 2000] if not SMOKE else [250, 1000]
for ns in sample_sizes:
    print(elapsed(), f"  Sample size sweep: num_samples={ns} (n={N_SENS})...")
    sens_results['sample_size'][ns] = run_lime_config(ns, K, True, N_SENS)
    print(elapsed(), f"    -> {sens_results['sample_size'][ns]}")

feature_counts = [5, 10, 15, 20] if not SMOKE else [5, 10]
for nf in feature_counts:
    print(elapsed(), f"  num_features sweep: K={nf} (n={N_SENS})...")
    sens_results['num_features'][nf] = run_lime_config(LIME_SAMPLES, nf, True, N_SENS)
    print(elapsed(), f"    -> {sens_results['num_features'][nf]}")

for disc in [True, False]:
    print(elapsed(), f"  Discretization sweep: discretize={disc} (n={N_SENS})...")
    sens_results['discretization'][str(disc)] = run_lime_config(LIME_SAMPLES, K, disc, N_SENS)
    print(elapsed(), f"    -> {sens_results['discretization'][str(disc)]}")

results['analysis_B_lime_sensitivity'] = sens_results
with open('supplemental_B_lime_sensitivity.json', 'w') as f:
    json.dump(sens_results, f, indent=2)
print(elapsed(), "  [CHECKPOINT] supplemental_B_lime_sensitivity.json saved")

# =============================================================================
# ANALYSIS C — raw per-instance Jaccard arrays + paired test
# =============================================================================
print(elapsed(), "=" * 60)
print(elapsed(), "ANALYSIS C: paired Jaccard consistency test")
print(elapsed(), "=" * 60)

print(elapsed(), "  SHAP consistency runs (deterministic, fast)...")
shap_runs = []
for run in range(N_RUNS):
    sv = explainer_default.shap_values(X_consist_raw)
    shap_runs.append([set(np.argsort(np.abs(sv[i]))[::-1][:K]) for i in range(N_CONSIST)])

print(elapsed(), "  LIME consistency runs (stochastic, slow)...")
lime_runs = []
for run in range(N_RUNS):
    print(elapsed(), f"    LIME run {run+1}/{N_RUNS}...")
    exp_i = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_filled.values, feature_names=feature_names,
        class_names=['Repaid', 'Default'], mode='classification',
        discretize_continuous=True, random_state=RANDOM_STATE)
    run_feats = []
    for i in range(N_CONSIST):
        try:
            exp = exp_i.explain_instance(
                data_row=X_consist_filled.iloc[i].values, predict_fn=predict_fn,
                num_features=K, num_samples=LIME_SAMPLES)
            run_feats.append(set(f[0] for f in exp.as_list()))
        except Exception:
            run_feats.append(set())
    lime_runs.append(run_feats)

# per-instance aggregation (mean Jaccard across the 45 run-pairs, per instance)
shap_per_instance = []
lime_per_instance = []
for s in range(N_CONSIST):
    shap_vals_s = [jaccard(shap_runs[r1][s], shap_runs[r2][s])
                   for r1, r2 in combinations(range(N_RUNS), 2)]
    lime_vals_s = [jaccard(lime_runs[r1][s], lime_runs[r2][s])
                   for r1, r2 in combinations(range(N_RUNS), 2)]
    shap_per_instance.append(float(np.mean(shap_vals_s)))
    lime_per_instance.append(float(np.mean(lime_vals_s)))

wilcoxon_stat, wilcoxon_p = stats.wilcoxon(shap_per_instance, lime_per_instance)
print(elapsed(), f"  Paired Wilcoxon (n={N_CONSIST} instances): "
                  f"stat={wilcoxon_stat:.1f}, p={wilcoxon_p:.3e}")
print(elapsed(), f"  SHAP per-instance mean J: {np.mean(shap_per_instance):.4f} "
                  f"± {np.std(shap_per_instance):.4f}")
print(elapsed(), f"  LIME per-instance mean J: {np.mean(lime_per_instance):.4f} "
                  f"± {np.std(lime_per_instance):.4f}")

results['analysis_C_paired_jaccard'] = {
    'shap_per_instance': shap_per_instance,
    'lime_per_instance': lime_per_instance,
    'wilcoxon_stat': float(wilcoxon_stat),
    'wilcoxon_p': float(wilcoxon_p),
    'n_instances': N_CONSIST,
}
with open('supplemental_C_paired_jaccard.json', 'w') as f:
    json.dump(results['analysis_C_paired_jaccard'], f, indent=2)
print(elapsed(), "  [CHECKPOINT] supplemental_C_paired_jaccard.json saved")

print(elapsed(), "ALL SUPPLEMENTAL ANALYSES COMPLETE.")
