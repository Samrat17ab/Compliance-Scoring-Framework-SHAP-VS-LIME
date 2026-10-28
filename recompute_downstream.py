#!/usr/bin/env python3
"""Recompute CSF scores, Monte Carlo weight sensitivity, and threshold
perturbation using the corrected probability-space C4 metric.
All other raw metrics are the PUBLISHED values (unaffected by this fix)."""
import json
import numpy as np

np.random.seed(42)

def score_coverage(v):
    if v >= 0.90: return 5.0
    if v >= 0.80: return 4.0
    if v >= 0.70: return 3.0
    if v >= 0.60: return 2.0
    return 1.0

def score_fidelity(v):
    if v >= 0.85: return 5.0
    if v >= 0.75: return 4.0
    if v >= 0.65: return 3.0
    if v >= 0.55: return 2.0
    return 1.5

def score_jaccard(v):
    if v >= 0.95: return 5.0
    if v >= 0.85: return 4.0
    if v >= 0.75: return 3.0
    if v >= 0.60: return 2.0
    return 1.0

def score_complexity(v):
    if v <= 0.40: return 5.0
    if v <= 0.55: return 4.0
    if v <= 0.70: return 3.0
    if v <= 0.85: return 2.0
    return 1.0

def score_bias_gap(v):
    if v >= 0.05:  return 5.0
    if v >= 0.03:  return 4.0
    if v >= 0.010: return 3.0
    if v >= 0.005: return 2.0
    return 1.0

def score_disparity(v):
    if v >= 0.05: return 5.0
    if v >= 0.03: return 4.0
    if v >= 0.01: return 3.0
    return 2.0

# Published raw metrics (unaffected by the C4 unit fix)
shap_c1_mean = 0.700
lime_c1_mean = 0.251
shap_c2_mean = 1.000
lime_c2_mean = 0.381
shap_c3a_mean = 0.536
lime_c3a_mean = 1.044
shap_c5_disp = 0.216
lime_c5_disp = 0.216
lime_c4_gap  = 0.007925   # unchanged, already probability-space

# CORRECTED: SHAP C4 now measured in probability space (exact interventional SHAP)
shap_c4_gap_prob = 0.038951
fold = shap_c4_gap_prob / lime_c4_gap

shap_scores = {
    'C1_Coverage':       score_coverage(shap_c1_mean),
    'C2_Consistency':    score_jaccard(shap_c2_mean),
    'C3a_Complexity':    score_complexity(shap_c3a_mean),
    'C3b_Comprehension': 2.0,
    'C4_Bias':           score_bias_gap(shap_c4_gap_prob),
    'C5_Fairness':       score_disparity(shap_c5_disp),
    'C6_Oversight':      4.0,
    'C7_Auditability':   5.0,
}
lime_scores = {
    'C1_Fidelity':       score_fidelity(lime_c1_mean),
    'C2_Consistency':    score_jaccard(lime_c2_mean),
    'C3a_Complexity':    score_complexity(lime_c3a_mean),
    'C3b_Comprehension': 4.0,
    'C4_Bias':           score_bias_gap(lime_c4_gap),
    'C5_Fairness':       score_disparity(lime_c5_disp),
    'C6_Oversight':      3.5,
    'C7_Auditability':   2.0,
}
shap_overall = float(np.mean(list(shap_scores.values())))
lime_overall = float(np.mean(list(lime_scores.values())))

print("CORRECTED CSF SCORES")
print(f"{'Criterion':<22}{'SHAP':>6}{'LIME':>6}{'Delta':>8}")
for (cs, sv), (cl, lv) in zip(shap_scores.items(), lime_scores.items()):
    print(f"{cs:<22}{sv:>6.1f}{lv:>6.1f}{sv-lv:>+8.1f}")
print(f"{'Overall':<22}{shap_overall:>6.2f}{lime_overall:>6.2f}{shap_overall-lime_overall:>+8.2f}")
print(f"\nC4 fold (probability space): {fold:.2f}x  (SHAP {shap_c4_gap_prob:.4f} vs LIME {lime_c4_gap:.4f})")

# ---- Monte Carlo weight sensitivity (corrected vectors) ----
shap_vector = np.array(list(shap_scores.values()))
lime_vector = np.array(list(lime_scores.values()))
n_sim = 3000
weights = np.random.dirichlet(np.ones(8), size=n_sim)
shap_wins = 0
score_differences = []
for w in weights:
    ss = np.dot(w, shap_vector)
    ls = np.dot(w, lime_vector)
    score_differences.append(ss - ls)
    if ss > ls:
        shap_wins += 1
robustness_ratio = shap_wins / n_sim
print(f"\nMonte Carlo (corrected): SHAP wins {shap_wins}/{n_sim} = {robustness_ratio:.4f}")

# which criterion drives LIME wins? check C3b weight for losing draws
lime_win_mask = np.array(score_differences) <= 0
if lime_win_mask.sum() > 0:
    c3b_weights_lime_wins = weights[lime_win_mask][:, 3]  # index 3 = C3b
    print(f"LIME-favouring cases: {lime_win_mask.sum()}, "
          f"C3b weight range: {c3b_weights_lime_wins.min():.3f}-{c3b_weights_lime_wins.max():.3f}, "
          f"min C3b weight among them: {c3b_weights_lime_wins.min():.3f}")

# ---- Threshold perturbation test (corrected C4 raw metric) ----
metrics_shap = {'C1': shap_c1_mean, 'C2': shap_c2_mean, 'C3a': shap_c3a_mean,
                 'C4': shap_c4_gap_prob, 'C5': shap_c5_disp}
metrics_lime = {'C1': lime_c1_mean, 'C2': lime_c2_mean, 'C3a': lime_c3a_mean,
                 'C4': lime_c4_gap, 'C5': lime_c5_disp}
perturbation = 0.10

def perturb(value, factor, direction):
    return value * (1 + factor) if direction == "up" else value * (1 - factor)

results = []
for direction in ["up", "down"]:
    s = score_coverage(perturb(shap_c1_mean, perturbation, direction))
    l = score_fidelity(perturb(lime_c1_mean, perturbation, direction))
    s += score_jaccard(perturb(shap_c2_mean, perturbation, direction))
    l += score_jaccard(perturb(lime_c2_mean, perturbation, direction))
    s += score_complexity(perturb(shap_c3a_mean, perturbation, direction))
    l += score_complexity(perturb(lime_c3a_mean, perturbation, direction))
    s += score_bias_gap(perturb(shap_c4_gap_prob, perturbation, direction))
    l += score_bias_gap(perturb(lime_c4_gap, perturbation, direction))
    s += score_disparity(perturb(shap_c5_disp, perturbation, direction))
    l += score_disparity(perturb(lime_c5_disp, perturbation, direction))
    results.append({'direction': direction, 'shap_score': s/5, 'lime_score': l/5, 'shap_wins': s > l})

print("\nThreshold perturbation (corrected):")
for r in results:
    print(r)

with open('supplemental_D_corrected_scores.json', 'w') as f:
    json.dump({
        'shap_scores': shap_scores, 'lime_scores': lime_scores,
        'shap_overall': shap_overall, 'lime_overall': lime_overall,
        'c4_fold': fold, 'c4_shap_prob_gap': shap_c4_gap_prob,
        'monte_carlo': {'shap_wins': shap_wins, 'n_sim': n_sim, 'robustness_ratio': robustness_ratio},
        'threshold_perturbation': results,
    }, f, indent=2)
print("\n[CHECKPOINT] supplemental_D_corrected_scores.json saved")
