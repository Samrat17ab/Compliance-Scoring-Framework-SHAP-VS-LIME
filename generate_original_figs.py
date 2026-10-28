#!/usr/bin/env python3
"""Regenerate Fig 3 (C4, original log-odds framing), Fig 4 (radar,
original scores), and Fig weight_sensitivity (original Monte Carlo)
to match the ORIGINAL submitted paper's numbers, since the minimal-diff
revision keeps C4 unrescored (SHAP: 5.0/5.0, LIME: 2.0/5.0 after fixing
only the Table V typo, not the units)."""
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from math import pi

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

shap_c4_gap = 0.2696
lime_c4_gap = 0.0079
fold = shap_c4_gap / lime_c4_gap  # ~34.1
lime_appear_pct = 7.5
SAMPLE_N = 1000

shap_scores = {
    'C1_Coverage': 3.0, 'C2_Consistency': 5.0, 'C3a_Complexity': 4.0,
    'C3b_Comprehension': 2.0, 'C4_Bias': 5.0, 'C5_Fairness': 5.0,
    'C6_Oversight': 4.0, 'C7_Auditability': 5.0,
}
lime_scores = {
    'C1_Fidelity': 1.5, 'C2_Consistency': 1.0, 'C3a_Complexity': 1.0,
    'C3b_Comprehension': 4.0, 'C4_Bias': 2.0, 'C5_Fairness': 5.0,
    'C6_Oversight': 3.5, 'C7_Auditability': 2.0,
}
shap_overall = float(np.mean(list(shap_scores.values())))  # 4.125 -> 4.12
lime_overall = float(np.mean(list(lime_scores.values())))  # 2.50

# ============================ FIG 3 ============================
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
fig.suptitle('Fig. 3. C4 Bias Detection', fontweight='bold', fontsize=11, y=0.98)

ax1 = axes[0]
gap_bars = ax1.bar(['SHAP', 'LIME'], [shap_c4_gap, lime_c4_gap],
                   color=[SHAP_COL, LIME_COL], width=0.45, edgecolor='white')
for i, (bar, val) in enumerate(zip(gap_bars, [shap_c4_gap, lime_c4_gap])):
    if i == 1:
        ax1.text(bar.get_x() + bar.get_width() / 2 - 0.25,
                 bar.get_height() + 0.006,
                 f'Δ={val:.4f}', ha='center', fontsize=8.5, fontweight='bold')
    else:
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.006,
                 f'Δ={val:.4f}', ha='center', fontsize=8.5, fontweight='bold')
ax1.annotate('', xy=(1, lime_c4_gap + 0.002), xytext=(0, shap_c4_gap - 0.002),
             arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.1))
ax1.text(0.5, (shap_c4_gap + lime_c4_gap) / 2 + 0.02,
         f'{fold:.0f}× gap', ha='center', va='center',
         fontsize=8, color='#555555', style='italic',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                   edgecolor='#CCCCCC', alpha=0.95))
ax1.set_xticks([0, 1])
ax1.set_xticklabels([
    f'SHAP\n(score: {shap_scores["C4_Bias"]:.1f}/5)',
    f'LIME\n(score: {lime_scores["C4_Bias"]:.1f}/5)'])
ax1.set_ylabel('|f̄_group1 - f̄_group2|  (signed means)', fontsize=8.5)
ax1.set_title(f'(A) Attribution Gap:\nSHAP Δ={shap_c4_gap:.4f} vs. LIME Δ={lime_c4_gap:.4f}',
              fontsize=9, pad=8)
ax1.set_ylim(0, shap_c4_gap * 1.35)

ax2 = axes[1]
lime_absent_pct = 100.0 - lime_appear_pct
ax2.bar([0], [100.0], color=SHAP_COL, width=0.45, edgecolor='white')
ax2.text(0, 50, f'100%\n(all {SAMPLE_N}\ninstances)',
         ha='center', va='center', color='white', fontsize=8.5, fontweight='bold')
ax2.bar([1], [lime_appear_pct], color=LIME_COL, width=0.45, edgecolor='white')
ax2.bar([1], [lime_absent_pct], bottom=[lime_appear_pct],
        color='#F5C6A0', width=0.45, edgecolor='white')
ax2.text(1, lime_appear_pct / 2, f'{lime_appear_pct:.1f}%\n(75/1000)',
         ha='center', va='center', color='white', fontsize=7.5, fontweight='bold')
ax2.text(1, lime_appear_pct + lime_absent_pct / 2,
         f'{lime_absent_pct:.1f}%\nNo signal',
         ha='center', va='center', color='#7B3200', fontsize=8.5, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['SHAP\n(unconditional)', 'LIME\n(selective)'])
ax2.set_ylabel('% of Instances', fontsize=8.5)
ax2.set_ylim(0, 125)
ax2.set_title(f'(B) Gender Appearance Rate:\nSHAP 100%, LIME {lime_appear_pct:.1f}%', fontsize=9, pad=8)
ax2.legend(handles=[
    mpatches.Patch(color=SHAP_COL, label='Gender attributed (SHAP)'),
    mpatches.Patch(color=LIME_COL, label='Gender in top-10 (LIME)'),
    mpatches.Patch(color='#F5C6A0', label='Gender absent'),
], fontsize=6.5, loc='upper right', framealpha=0.95)
plt.tight_layout()
plt.savefig('fig3_bias_detection.png', dpi=300, bbox_inches='tight')
plt.savefig('fig3_bias_detection.pdf', bbox_inches='tight')
plt.close()
print("fig3_bias_detection.png/.pdf saved (original numbers)")

# ============================ FIG 4 ============================
radar_labels = ['C1\nCoverage', 'C2\nConsistency', 'C3a\nComplexity',
                 'C3b\nComprehension', 'C4\nBias', 'C5\nFairness',
                 'C6\nOversight', 'C7\nAuditability']
N = len(radar_labels)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
s_vals = list(shap_scores.values()) + [list(shap_scores.values())[0]]
l_vals = list(lime_scores.values()) + [list(lime_scores.values())[0]]

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('white')
ax.plot(angles, s_vals, 'o-', linewidth=2.2, color=SHAP_COL,
        label=f'SHAP ({shap_overall:.2f}/5.00)', markersize=6)
ax.fill(angles, s_vals, alpha=0.18, color=SHAP_COL)
ax.plot(angles, l_vals, 's-', linewidth=2.2, color=LIME_COL,
        label=f'LIME ({lime_overall:.2f}/5.00)', markersize=6)
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
ax.set_title(
    'Fig. 4. CSF Compliance Radar Chart\n'
    'SHAP dominates regulatory dimensions; LIME leads on user comprehensibility (C3b)',
    fontsize=10, fontweight='bold', pad=28)
ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.08), fontsize=9, framealpha=0.95)
plt.tight_layout()
plt.savefig('fig4_radar_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('fig4_radar_chart.pdf', bbox_inches='tight')
plt.close()
print("fig4_radar_chart.png/.pdf saved (original numbers)")

# ============================ FIG weight_sensitivity ============================
np.random.seed(42)
shap_vector = np.array(list(shap_scores.values()))
lime_vector = np.array(list(lime_scores.values()))
n_sim = 3000
weights = np.random.dirichlet(np.ones(8), size=n_sim)
score_differences = weights @ shap_vector - weights @ lime_vector
shap_wins = int((score_differences > 0).sum())
print(f"Monte Carlo check: SHAP wins {shap_wins}/{n_sim} = {shap_wins/n_sim:.4f} "
      f"(paper claims 2971/3000 = 0.9903)")

plt.figure(figsize=(6, 4))
plt.hist(score_differences, bins=40, edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.title('Distribution of (SHAP score - LIME score)')
plt.xlabel('Score Difference')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('fig_weight_sensitivity.png', dpi=300)
plt.close()
print("fig_weight_sensitivity.png saved (original numbers)")
