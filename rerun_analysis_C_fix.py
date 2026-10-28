#!/usr/bin/env python3
"""
Fix for Analysis C: the LIME explainer must be created ONCE and reused
across all N_RUNS stochastic runs (matching csf_evaluation.py Step 5's
C2 stochastic test), NOT recreated with the same seed every run (that
reproduces the *fixed-seed* experiment instead, giving trivial J=1.000).
"""
import json
import time
import warnings
from itertools import combinations

import lightgbm as lgb
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
np.random.seed(42)

SAMPLE_N     = 1000
N_CONSIST    = 100
N_RUNS       = 10
K            = 10
LIME_SAMPLES = 1000
RANDOM_STATE = 42
GENDER_COL   = 'CODE_GENDER'

t0 = time.time()
def elapsed():
    return f"[{time.time()-t0:6.1f}s]"

def jaccard(s1, s2):
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)

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

print(elapsed(), "Training LightGBM...")
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

X_sample_filled = X_test_filled.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
X_sample_raw    = X_test.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
X_consist_filled = X_sample_filled.iloc[:N_CONSIST].copy()
X_consist_raw     = X_sample_raw.iloc[:N_CONSIST].copy()

def predict_fn(x):
    d = pd.DataFrame(x, columns=feature_names)
    p = model.predict(d)
    return np.column_stack([1 - p, p])

import shap
explainer_default = shap.TreeExplainer(model)

print(elapsed(), "SHAP consistency runs (deterministic)...")
shap_runs = []
for run in range(N_RUNS):
    sv = explainer_default.shap_values(X_consist_raw)
    shap_runs.append([set(np.argsort(np.abs(sv[i]))[::-1][:K]) for i in range(N_CONSIST)])

print(elapsed(), "LIME STOCHASTIC consistency runs "
                  "(single explainer instance reused across all runs, per csf_evaluation.py Step 5)...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_filled.values, feature_names=feature_names,
    class_names=['Repaid', 'Default'], mode='classification',
    discretize_continuous=True, random_state=RANDOM_STATE)

lime_runs = []
for run in range(N_RUNS):
    print(elapsed(), f"  LIME stochastic run {run+1}/{N_RUNS}...")
    run_feats = []
    for i in range(N_CONSIST):
        try:
            exp = lime_explainer.explain_instance(
                data_row=X_consist_filled.iloc[i].values, predict_fn=predict_fn,
                num_features=K, num_samples=LIME_SAMPLES)
            run_feats.append(set(f[0] for f in exp.as_list()))
        except Exception:
            run_feats.append(set())
    lime_runs.append(run_feats)

# sanity check against aggregate stats (should be close to published 0.381 +/- 0.114)
all_lime_pairs = [jaccard(lime_runs[r1][s], lime_runs[r2][s])
                   for r1, r2 in combinations(range(N_RUNS), 2)
                   for s in range(N_CONSIST)]
print(elapsed(), f"  Aggregate stochastic LIME J (all 4500 pairs): "
                  f"{np.mean(all_lime_pairs):.4f} +/- {np.std(all_lime_pairs):.4f}  "
                  f"(published: 0.381 +/- 0.114)")

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
print(elapsed(), f"Paired Wilcoxon (n={N_CONSIST} instances): stat={wilcoxon_stat:.1f}, p={wilcoxon_p:.3e}")
print(elapsed(), f"SHAP per-instance mean J: {np.mean(shap_per_instance):.4f} +/- {np.std(shap_per_instance):.4f}")
print(elapsed(), f"LIME per-instance mean J: {np.mean(lime_per_instance):.4f} +/- {np.std(lime_per_instance):.4f}")

with open('supplemental_C_paired_jaccard.json', 'w') as f:
    json.dump({
        'shap_per_instance': shap_per_instance,
        'lime_per_instance': lime_per_instance,
        'wilcoxon_stat': float(wilcoxon_stat),
        'wilcoxon_p': float(wilcoxon_p),
        'n_instances': N_CONSIST,
        'aggregate_lime_mean': float(np.mean(all_lime_pairs)),
        'aggregate_lime_std': float(np.std(all_lime_pairs)),
    }, f, indent=2)
print(elapsed(), "[CHECKPOINT] supplemental_C_paired_jaccard.json overwritten with corrected results")
