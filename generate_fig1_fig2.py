#!/usr/bin/env python3
"""Generate the two remaining missing standard figures (Fig 1 kernel
sweep, Fig 2 consistency) so the paper's \\includegraphics references
resolve to real files. Does not change any paper claims/numbers."""
import time
import warnings

import lightgbm as lgb
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')
np.random.seed(42)

SAMPLE_N = 1000
SWEEP_N = 200
N_CONSIST = 100
N_RUNS = 10
K = 10
LIME_SAMPLES = 1000
RANDOM_STATE = 42
KERNEL_MULTIPLIERS = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0]

t0 = time.time()
def elapsed():
    return f"[{time.time()-t0:6.1f}s]"

def jaccard(s1, s2):
    union = s1 | s2
    if not union:
        return 1.0
    return len(s1 & s2) / len(union)

print(elapsed(), "Loading data & training LightGBM...")
df = pd.read_csv('application_train.csv')
y = df['TARGET']
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
missing_frac = X.isnull().sum() / len(X)
X = X.drop(missing_frac[missing_frac == 1.0].index.tolist(), axis=1)
for col in X.select_dtypes(include='object').columns:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=RANDOM_STATE, stratify=y_temp)
train_medians = X_train.median(numeric_only=True)
X_train_filled = X_train.fillna(train_medians)
X_test_filled = X_test.fillna(train_medians)
feature_names = list(X_train.columns)

scale_pw = (y_train == 0).sum() / (y_train == 1).sum()
lgb_train_ds = lgb.Dataset(X_train, label=y_train)
lgb_val_ds = lgb.Dataset(X_val, label=y_val, reference=lgb_train_ds)
lgb_params = {'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt', 'num_leaves': 31,
              'learning_rate': 0.05, 'feature_fraction': 0.9, 'bagging_fraction': 0.8,
              'bagging_freq': 5, 'scale_pos_weight': scale_pw, 'verbose': -1, 'random_state': RANDOM_STATE}
model = lgb.train(lgb_params, lgb_train_ds, num_boost_round=1000, valid_sets=[lgb_val_ds],
                   callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

X_sample_raw = X_test.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
X_sample_filled = X_test_filled.sample(SAMPLE_N, random_state=RANDOM_STATE).reset_index(drop=True)
X_consist_raw = X_sample_raw.iloc[:N_CONSIST].copy()
X_consist_filled = X_sample_filled.iloc[:N_CONSIST].copy()

def predict_fn(x):
    d = pd.DataFrame(x, columns=feature_names)
    p = model.predict(d)
    return np.column_stack([1 - p, p])

SHAP_COL = '#1F4E79'
LIME_COL = '#C55A11'
FIXED_COL = '#2E75B6'
GRID_COL = '#E8E8E8'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': GRID_COL, 'grid.linewidth': 0.7,
})

# ============================ FIG 1 — kernel width sweep ============================
n_features = X_train_filled.shape[1]
default_kw = np.sqrt(n_features) * 0.75
print(elapsed(), f"Default kernel width: {default_kw:.4f}")

kernel_results = {}
for mult in KERNEL_MULTIPLIERS:
    kw = default_kw * mult
    print(elapsed(), f"  Kernel x{mult} (width={kw:.3f})...")
    sweep_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_filled.values, feature_names=feature_names,
        class_names=['Repaid', 'Default'], mode='classification',
        discretize_continuous=True, kernel_width=kw, random_state=RANDOM_STATE)
    r2_run = []
    for i in range(SWEEP_N):
        try:
            exp = sweep_exp.explain_instance(data_row=X_sample_filled.iloc[i].values,
                                              predict_fn=predict_fn, num_features=K, num_samples=LIME_SAMPLES)
            r2_run.append(float(exp.score))
        except Exception:
            pass
    kernel_results[mult] = {'kernel_width': float(kw), 'r2_mean': float(np.mean(r2_run)), 'r2_std': float(np.std(r2_run))}
    print(elapsed(), f"    R2 = {np.mean(r2_run):.4f} +/- {np.std(r2_run):.4f}")

km = [float(k) for k in kernel_results.keys()]
rm = [kernel_results[k]['r2_mean'] for k in kernel_results]
rs = [kernel_results[k]['r2_std'] for k in kernel_results]

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(km, rm, 'o-', color=LIME_COL, linewidth=2.2, markersize=6, markerfacecolor=LIME_COL,
        markeredgecolor='white', markeredgewidth=1.2, zorder=3)
ax.fill_between(km, [m - s for m, s in zip(rm, rs)], [m + s for m, s in zip(rm, rs)],
                 color=LIME_COL, alpha=0.15, zorder=2)
di = km.index(1.0)
ax.scatter([km[di]], [rm[di]], s=90, zorder=5, color=LIME_COL, edgecolors='black', linewidths=1.2)
ax.annotate(f'Default\nR2={rm[di]:.4f}', xy=(km[di], rm[di]), xytext=(1.35, 0.282),
            fontsize=8, color='#444444', arrowprops=dict(arrowstyle='->', color='#888888', lw=0.9), ha='left')
for m, r in zip(km, rm):
    ax.text(m, r + 0.016, f'{r:.4f}', ha='center', fontsize=7.5, color=LIME_COL, fontweight='bold')
r_min, r_max = min(rm), max(rm)
ax.annotate('', xy=(km[-1], r_max + 0.055), xytext=(km[0], r_max + 0.055),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0))
ax.text(np.mean([km[0], km[-1]]), r_max + 0.068,
        f'Range {r_min:.4f}-{r_max:.4f} across {int(km[-1]/km[0])}x kernel range',
        ha='center', fontsize=8, color='#444444', style='italic')
ax.set_xlabel('Kernel Width Multiplier (x default)', fontsize=9)
ax.set_ylabel('Local Fidelity R2', fontsize=9)
ax.set_title(f'Fig. 1. Kernel Width Sensitivity Analysis (n={SWEEP_N} per config)', fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(km)
ax.set_xticklabels([f'x{m}' for m in km])
ax.set_ylim(0.24, r_max + 0.11)
ax.text(0.02, 0.96, 'Score-2 (R2>=0.55) and Score-3 (R2>=0.65)\nthresholds lie far above this range',
        transform=ax.transAxes, fontsize=7.5, color='#888888', va='top', ha='left', style='italic')
plt.tight_layout()
plt.savefig('fig1_kernel_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig('fig1_kernel_sweep.pdf', bbox_inches='tight')
plt.close()
print(elapsed(), "fig1_kernel_sweep.png/.pdf saved")

# ============================ FIG 2 — consistency (SHAP, LIME fixed-seed, LIME stochastic) ============================
explainer = shap.TreeExplainer(model)
print(elapsed(), "SHAP consistency (deterministic)...")
shap_runs = []
for run in range(N_RUNS):
    sv = explainer.shap_values(X_consist_raw)
    shap_runs.append([set(np.argsort(np.abs(sv[i]))[::-1][:K]) for i in range(N_CONSIST)])
shap_j = [jaccard(shap_runs[r1][s], shap_runs[r2][s]) for r1, r2 in combinations(range(N_RUNS), 2) for s in range(N_CONSIST)]
shap_c2_mean, shap_c2_std = float(np.mean(shap_j)), float(np.std(shap_j))

print(elapsed(), "LIME fixed-seed consistency (fresh explainer, same seed, each run)...")
fixed_runs = []
for run in range(N_RUNS):
    print(elapsed(), f"  fixed-seed run {run+1}/{N_RUNS}...")
    fresh_exp = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_filled.values, feature_names=feature_names,
        class_names=['Repaid', 'Default'], mode='classification',
        discretize_continuous=True, random_state=RANDOM_STATE)
    run_feats = []
    for i in range(N_CONSIST):
        try:
            exp = fresh_exp.explain_instance(data_row=X_consist_filled.iloc[i].values, predict_fn=predict_fn,
                                              num_features=K, num_samples=LIME_SAMPLES)
            run_feats.append(set(f[0] for f in exp.as_list()))
        except Exception:
            run_feats.append(set())
    fixed_runs.append(run_feats)
fixed_j = [jaccard(fixed_runs[r1][s], fixed_runs[r2][s]) for r1, r2 in combinations(range(N_RUNS), 2) for s in range(N_CONSIST) if fixed_runs[r1][s] or fixed_runs[r2][s]]
fixed_c2_mean, fixed_c2_std = float(np.mean(fixed_j)), float(np.std(fixed_j))
print(elapsed(), f"Fixed-seed LIME: {fixed_c2_mean:.4f} +/- {fixed_c2_std:.4f}")

print(elapsed(), "LIME stochastic consistency (single explainer, reused across runs)...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_filled.values, feature_names=feature_names,
    class_names=['Repaid', 'Default'], mode='classification',
    discretize_continuous=True, random_state=RANDOM_STATE)
lime_runs = []
for run in range(N_RUNS):
    print(elapsed(), f"  stochastic run {run+1}/{N_RUNS}...")
    run_feats = []
    for i in range(N_CONSIST):
        try:
            exp = lime_explainer.explain_instance(data_row=X_consist_filled.iloc[i].values, predict_fn=predict_fn,
                                                    num_features=K, num_samples=LIME_SAMPLES)
            run_feats.append(set(f[0] for f in exp.as_list()))
        except Exception:
            run_feats.append(set())
    lime_runs.append(run_feats)
lime_j = [jaccard(lime_runs[r1][s], lime_runs[r2][s]) for r1, r2 in combinations(range(N_RUNS), 2) for s in range(N_CONSIST) if lime_runs[r1][s] or lime_runs[r2][s]]
lime_c2_mean, lime_c2_std = float(np.mean(lime_j)), float(np.std(lime_j))
print(elapsed(), f"Stochastic LIME: {lime_c2_mean:.4f} +/- {lime_c2_std:.4f}  (published: 0.381 +/- 0.114)")

fig, ax = plt.subplots(figsize=(6, 4))
cats = ['SHAP\n(deterministic)', 'LIME\n(fixed seed)', 'LIME\n(stochastic)']
means_bar = [shap_c2_mean, fixed_c2_mean, lime_c2_mean]
stds_bar = [shap_c2_std, fixed_c2_std, lime_c2_std]
cols_bar = [SHAP_COL, FIXED_COL, LIME_COL]
bars = ax.bar(cats, means_bar, color=cols_bar, width=0.5, edgecolor='white', zorder=3)
ax.errorbar(range(3), means_bar, yerr=stds_bar, fmt='none', color='#555555', capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
max_err = max(stds_bar)
for bar, val in zip(bars, means_bar):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_err + 0.022,
            f'J = {val:.3f}', ha='center', fontsize=9, fontweight='bold')
ax.axhline(0.95, color='green', linestyle='--', linewidth=1.2, alpha=0.8, label='Score-5 (J=0.95)', zorder=2)
ax.axhline(0.75, color='orange', linestyle=':', linewidth=1.2, alpha=0.8, label='Score-3 (J=0.75)', zorder=2)
ax.annotate('Requires per-instance\nseed management', xy=(1, fixed_c2_mean), xytext=(1.55, fixed_c2_mean + 0.18),
            fontsize=7.5, color=FIXED_COL, style='italic', arrowprops=dict(arrowstyle='->', color=FIXED_COL, lw=0.9), ha='center')
ax.set_ylabel('Jaccard Similarity (J)', fontsize=9)
ax.set_ylim(0, 1.25)
ax.set_title(f'Fig. 2. C2+C7 Consistency: Jaccard Similarity\n'
             f'SHAP: J={shap_c2_mean:.3f}, Fixed-seed LIME: J={fixed_c2_mean:.3f}, '
             f'Stochastic LIME: J={lime_c2_mean:.3f}+/-{lime_c2_std:.3f}', fontsize=9.5, fontweight='bold', pad=10)
ax.legend(fontsize=7.5, loc='upper left', framealpha=0.9)
plt.tight_layout()
plt.savefig('fig2_consistency.png', dpi=300, bbox_inches='tight')
plt.savefig('fig2_consistency.pdf', bbox_inches='tight')
plt.close()
print(elapsed(), "fig2_consistency.png/.pdf saved")
print(elapsed(), "DONE.")
