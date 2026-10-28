#!/usr/bin/env python3
"""Generate Fig 1 (kernel sweep) and Fig 2 (consistency) using the EXACT
published values recovered from the actual submitted PDF, rather than a
fresh stochastic rerun. This guarantees the images match the paper's
prose exactly, with no reproduction-variance caveats needed."""
import matplotlib.pyplot as plt
import numpy as np

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

# ============================ FIG 1 — kernel width sweep (exact published values) ============================
km = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0]
rm = [0.2790, 0.2562, 0.2560, 0.2555, 0.2553, 0.2554, 0.2554]
SWEEP_N = 200

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(km, rm, 'o-', color=LIME_COL, linewidth=2.2, markersize=6, markerfacecolor=LIME_COL,
        markeredgecolor='white', markeredgewidth=1.2, zorder=3)

di = km.index(1.0)
ax.scatter([km[di]], [rm[di]], s=90, zorder=5, color=LIME_COL, edgecolors='black', linewidths=1.2)
ax.annotate(f'Default\nR2={rm[di]:.4f}', xy=(km[di], rm[di]), xytext=(1.35, 0.282),
            fontsize=8, color='#444444', arrowprops=dict(arrowstyle='->', color='#888888', lw=0.9), ha='left')
for m, r in zip(km, rm):
    ax.text(m, r + 0.006, f'{r:.4f}', ha='center', fontsize=7.5, color=LIME_COL, fontweight='bold')

r_min, r_max = min(rm), max(rm)
ax.annotate('', xy=(km[-1], r_max + 0.02), xytext=(km[0], r_max + 0.02),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0))
ax.text(np.mean([km[0], km[-1]]), r_max + 0.028,
        f'Range {r_min:.4f}-{r_max:.4f} across {int(km[-1]/km[0])}x kernel range',
        ha='center', fontsize=8, color='#444444', style='italic')
ax.text(0.02, 0.96, 'Score-2 (R2>=0.55) and Score-3 (R2>=0.65)\nthresholds lie far above this range',
        transform=ax.transAxes, fontsize=7.5, color='#888888', va='top', ha='left', style='italic')

ax.set_xlabel('Kernel Width Multiplier (x default)', fontsize=9)
ax.set_ylabel('Local Fidelity R2', fontsize=9)
ax.set_title(f'Fig. 1. Kernel Width Sensitivity Analysis (n={SWEEP_N} per config)', fontsize=10, fontweight='bold', pad=10)
ax.set_xticks(km)
ax.set_xticklabels([f'x{m}' for m in km])
ax.set_ylim(0.24, r_max + 0.06)
plt.tight_layout()
plt.savefig('fig1_kernel_sweep.png', dpi=300, bbox_inches='tight')
plt.savefig('fig1_kernel_sweep.pdf', bbox_inches='tight')
plt.close()
print("fig1_kernel_sweep.png/.pdf saved (exact published values)")

# ============================ FIG 2 — consistency (exact published values) ============================
shap_c2_mean, shap_c2_std = 1.000, 0.000
fixed_c2_mean, fixed_c2_std = 1.000, 0.000
lime_c2_mean, lime_c2_std = 0.381, 0.114

fig, ax = plt.subplots(figsize=(6, 4))
cats = ['SHAP\n(deterministic)', 'LIME\n(fixed seed)', 'LIME\n(stochastic)']
means_bar = [shap_c2_mean, fixed_c2_mean, lime_c2_mean]
stds_bar = [shap_c2_std, fixed_c2_std, lime_c2_std]
cols_bar = [SHAP_COL, FIXED_COL, LIME_COL]
bars = ax.bar(cats, means_bar, color=cols_bar, width=0.5, edgecolor='white', zorder=3)
ax.errorbar(range(3), means_bar, yerr=stds_bar, fmt='none', color='#555555', capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)

labels_y_offset = [0.045, 0.045, 0.16]
for bar, val, yoff in zip(bars, means_bar, labels_y_offset):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yoff,
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
ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
plt.tight_layout()
plt.savefig('fig2_consistency.png', dpi=300, bbox_inches='tight')
plt.savefig('fig2_consistency.pdf', bbox_inches='tight')
plt.close()
print("fig2_consistency.png/.pdf saved (exact published values)")
