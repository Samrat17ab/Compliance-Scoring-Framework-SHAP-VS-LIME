#!/usr/bin/env python3
"""Regenerate Fig 3 (C4, corrected to probability space), Fig 4 (radar,
corrected scores), Fig weight_sensitivity (corrected Monte Carlo), and a
new Fig 5 (LIME parameter sensitivity) for reviewer points 4 and 5."""
import json
from math import pi

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

with open('supplemental_A_c4_probability.json') as f:
    A = json.load(f)
with open('supplemental_B_lime_sensitivity.json') as f:
    B = json.load(f)
with open('supplemental_D_corrected_scores.json') as f:
    D = json.load(f)

SHAP_COL = '#1F4E79'
LIME_COL = '#C55A11'
GRID_COL = '#E8E8E8'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'axes.labelsize': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': GRID_COL, 'grid.linewidth': 0.7,
})

shap_c4_gap = A['exact_interventional_prob_gap']
lime_c4_gap = 0.007925
fold = shap_c4_gap / lime_c4_gap
lime_appear_pct = 7.5
SAMPLE_N = 1000
gender_appearances = 75
n_lime = 1000
shap_scores = D['shap_scores']
lime_scores = D['lime_scores']

# ============================ FIG 3 (corrected) ============================
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
fig.suptitle('Fig. 3. C4 Bias Detection (probability-space, unit-corrected)',
             fontweight='bold', fontsize=10.5, y=0.98)

ax1 = axes[0]
gap_bars = ax1.bar(['SHAP\n(probability-space)', 'LIME'], [shap_c4_gap, lime_c4_gap],
                    color=[SHAP_COL, LIME_COL], width=0.45, edgecolor='white')
for i, (bar, val) in enumerate(zip(gap_bars, [shap_c4_gap, lime_c4_gap])):
    off = -0.20 if i == 1 else 0.0
    ax1.text(bar.get_x() + bar.get_width()/2 + off, bar.get_height() + 0.0015,
              f'{"Δ"}={val:.4f}', ha='center', fontsize=8.5, fontweight='bold')
ax1.annotate('', xy=(1, lime_c4_gap + 0.0006), xytext=(0, shap_c4_gap - 0.0006),
             arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.1))
ax1.text(0.5, (shap_c4_gap + lime_c4_gap)/2 + 0.006, f'{fold:.1f}× gap',
          ha='center', va='center', fontsize=8, color='#555555', style='italic',
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#CCCCCC', alpha=0.95))
ax1.set_xticks([0, 1])
ax1.set_xticklabels([f'SHAP\n(score: {shap_scores["C4_Bias"]:.1f}/5)',
                      f'LIME\n(score: {lime_scores["C4_Bias"]:.1f}/5)'])
ax1.set_ylabel('|mean prob.-space attribution gap|', fontsize=8)
ax1.set_title(f'(A) Attribution Gap (probability units):\nSHAP {"Δ"}={shap_c4_gap:.4f} vs. LIME {"Δ"}={lime_c4_gap:.4f}',
              fontsize=8.8, pad=8)
ax1.set_ylim(0, shap_c4_gap * 1.35)

ax2 = axes[1]
lime_absent_pct = 100.0 - lime_appear_pct
ax2.bar([0], [100.0], color=SHAP_COL, width=0.45, edgecolor='white')
ax2.text(0, 50, f'100%\n(all {SAMPLE_N}\ninstances)', ha='center', va='center',
          color='white', fontsize=8.5, fontweight='bold')
ax2.bar([1], [lime_appear_pct], color=LIME_COL, width=0.45, edgecolor='white')
ax2.bar([1], [lime_absent_pct], bottom=[lime_appear_pct], color='#F5C6A0', width=0.45, edgecolor='white')
ax2.text(1, lime_appear_pct + lime_absent_pct/2, f'{lime_absent_pct:.1f}%\nNo signal',
          ha='center', va='center', color='#7B3200', fontsize=8.5, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['SHAP\n(unconditional)', 'LIME\n(selective, K=10)'])
ax2.set_ylabel('% of Instances', fontsize=8.5)
ax2.set_ylim(0, 125)
ax2.set_title(f'(B) Gender Appearance Rate:\nSHAP 100%, LIME {lime_appear_pct:.1f}%', fontsize=8.8, pad=8)
ax2.legend(handles=[
    mpatches.Patch(color=SHAP_COL, label='Gender attributed (SHAP)'),
    mpatches.Patch(color=LIME_COL, label='Gender in top-10 (LIME)'),
    mpatches.Patch(color='#F5C6A0', label='Gender absent'),
], fontsize=6.5, loc='upper right', framealpha=0.95)
plt.tight_layout()
plt.savefig('fig3_bias_detection.png', dpi=300, bbox_inches='tight')
plt.savefig('fig3_bias_detection.pdf', bbox_inches='tight')
plt.close()
print("fig3_bias_detection.png/.pdf saved (corrected)")

# ============================ FIG 4 (corrected radar) ============================
radar_labels = ['C1\nCoverage', 'C2\nConsistency', 'C3a\nComplexity',
                 'C3b\nComprehension', 'C4\nBias', 'C5\nFairness',
                 'C6\nOversight', 'C7\nAuditability']
N = len(radar_labels)
angles = [n/float(N)*2*pi for n in range(N)]
angles += angles[:1]
s_vals = list(shap_scores.values()) + [list(shap_scores.values())[0]]
l_vals = list(lime_scores.values()) + [list(lime_scores.values())[0]]

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')
ax.plot(angles, s_vals, 'o-', linewidth=2.2, color=SHAP_COL,
        label=f'SHAP ({D["shap_overall"]:.2f}/5.00)', markersize=6)
ax.fill(angles, s_vals, alpha=0.18, color=SHAP_COL)
ax.plot(angles, l_vals, 's-', linewidth=2.2, color=LIME_COL,
        label=f'LIME ({D["lime_overall"]:.2f}/5.00)', markersize=6)
ax.fill(angles, l_vals, alpha=0.18, color=LIME_COL)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_labels, fontsize=9)
tick_labels = ax.get_xticklabels()
tick_labels[0].set_ha('left')
tick_labels[1].set_ha('left')
tick_labels[3].set_ha('right')
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=7.5)
ax.set_title('Fig. 4. CSF Compliance Radar Chart (corrected)\n'
             'SHAP dominates regulatory dimensions; LIME leads on user comprehensibility (C3b)',
             fontsize=10, fontweight='bold', pad=28)
ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08), fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('fig4_radar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('fig4_radar_chart.pdf', bbox_inches='tight')
plt.close()
print("fig4_radar_chart.png/.pdf saved (corrected)")

# ============================ FIG weight_sensitivity (corrected) ============================
np.random.seed(42)
shap_vector = np.array(list(shap_scores.values()))
lime_vector = np.array(list(lime_scores.values()))
n_sim = 3000
weights = np.random.dirichlet(np.ones(8), size=n_sim)
score_differences = weights @ shap_vector - weights @ lime_vector

plt.figure(figsize=(6, 4))
plt.hist(score_differences, bins=40, edgecolor='black', color=SHAP_COL, alpha=0.85)
plt.axvline(0, color='red', linestyle='--')
plt.title('Distribution of (SHAP score - LIME score)\n(corrected: probability-space C4)')
plt.xlabel('Score Difference')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('fig_weight_sensitivity.png', dpi=300)
plt.close()
print("fig_weight_sensitivity.png saved (corrected)")

# ============================ FIG 5 (new: LIME parameter sensitivity) ============================
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
fig.suptitle('Fig. 5. LIME Parameter Sensitivity (n=150 per configuration)',
             fontweight='bold', fontsize=11, y=1.03)

ax = axes[0]
sizes = sorted(int(k) for k in B['sample_size'].keys())
r2s = [B['sample_size'][str(s)]['r2_mean'] for s in sizes]
gr  = [B['sample_size'][str(s)]['gender_appear_rate']*100 for s in sizes]
ax2r = ax.twinx()
ax.plot(sizes, r2s, 'o-', color=LIME_COL, label='Fidelity $R^2$')
ax2r.plot(sizes, gr, 's--', color='#666666', label='Gender appear %')
ax.set_xlabel('num_samples (perturbations)')
ax.set_ylabel('Fidelity $R^2$', color=LIME_COL)
ax2r.set_ylabel('Gender appear. rate (%)', color='#666666')
ax.set_title('(A) Perturbation Sample Size', fontsize=9.5)

ax = axes[1]
ks = sorted(int(k) for k in B['num_features'].keys())
r2s = [B['num_features'][str(k)]['r2_mean'] for k in ks]
gr  = [B['num_features'][str(k)]['gender_appear_rate']*100 for k in ks]
ax2r = ax.twinx()
ax.plot(ks, r2s, 'o-', color=LIME_COL, label='Fidelity $R^2$')
ax2r.plot(ks, gr, 's--', color='#666666', label='Gender appear %')
ax.axvline(10, color='#999999', linestyle=':', linewidth=1)
ax.text(10.3, min(r2s), 'study\ndefault', fontsize=7, color='#666666')
ax.set_xlabel('num_features (K)')
ax.set_ylabel('Fidelity $R^2$', color=LIME_COL)
ax2r.set_ylabel('Gender appear. rate (%)', color='#666666')
ax.set_title('(B) Number of Reported Features (K)', fontsize=9.5)

ax = axes[2]
cats = ['Discretized\n(study default)', 'Continuous\n(raw features)']
r2s = [B['discretization']['True']['r2_mean'], B['discretization']['False']['r2_mean']]
gr  = [B['discretization']['True']['gender_appear_rate']*100, B['discretization']['False']['gender_appear_rate']*100]
sc  = [B['discretization']['True']['complexity_mean'], B['discretization']['False']['complexity_mean']]
x = np.arange(2)
w = 0.25
ax.bar(x - w, r2s, width=w, color=LIME_COL, label='Fidelity $R^2$')
ax.bar(x, [g/100 for g in gr], width=w, color='#666666', label='Gender appear. rate')
ax.bar(x + w, sc, width=w, color='#999999', label='Complexity (SC)')
ax.set_xticks(x)
ax.set_xticklabels(cats, fontsize=8)
ax.set_ylim(0, 1.1)
ax.legend(fontsize=6.5, loc='upper left')
ax.set_title('(C) Discretization Scheme', fontsize=9.5)

plt.tight_layout()
plt.savefig('fig5_lime_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('fig5_lime_sensitivity.pdf', bbox_inches='tight')
plt.close()
print("fig5_lime_sensitivity.png/.pdf saved (new)")
