#!/usr/bin/env python3
"""Threshold-sensitivity ablation for C1 specifically (Option 3):
shift the Coverage (SHAP) and Fidelity R^2 (LIME) score bands by a
range of percentages and check whether the SHAP > LIME C1 ranking
survives, and find the exact break-even shift for each side."""

shap_c1 = 0.700   # SHAP mean coverage
lime_c1 = 0.251   # LIME mean fidelity R^2

# Original Table IV thresholds
coverage_bands = [(0.90, 5.0), (0.80, 4.0), (0.70, 3.0), (0.60, 2.0)]  # else 1.0
r2_bands       = [(0.85, 5.0), (0.75, 4.0), (0.65, 3.0), (0.55, 2.0)]  # else 1.5 (LIME floor)

def score_with_shift(bands, value, shift, floor_score):
    """shift is a fractional multiplier applied to every cutoff, e.g. -0.10 = cutoffs lowered 10%."""
    for cutoff, score in bands:
        if value >= cutoff * (1 + shift):
            return score
    return floor_score

print(f"SHAP C1 raw = {shap_c1}, LIME C1 raw = {lime_c1}")
print(f"{'shift':>8} {'SHAP score':>11} {'LIME score':>11} {'SHAP>LIME':>10}")
for shift_pct in [-40, -30, -20, -15, -10, -5, -1, 0, 1, 5, 10, 15, 20, 30, 40]:
    shift = shift_pct / 100.0
    s = score_with_shift(coverage_bands, shap_c1, shift, 1.0)
    l = score_with_shift(r2_bands, lime_c1, shift, 1.5)
    print(f"{shift_pct:>7}% {s:>11.1f} {l:>11.1f} {str(s > l):>10}")

# Find exact break-even shift for SHAP's coverage score to drop from 3->2
# cutoff*  (1+shift) = value  =>  shift = value/cutoff - 1
shap_break_3to2 = shap_c1 / 0.70 - 1
print(f"\nSHAP coverage score drops from 3.0->2.0 at shift = {shap_break_3to2*100:.2f}%")
shap_break_2to1 = shap_c1 / 0.60 - 1
print(f"SHAP coverage score drops from 2.0->1.0 at shift = {shap_break_2to1*100:.2f}%")

# Find exact break-even shift for LIME's R^2 score to rise from 1.5->2.0
lime_break_to2 = lime_c1 / 0.55 - 1
print(f"LIME R^2 score would rise from 1.5->2.0 only if the Score-2 cutoff shifted by {lime_break_to2*100:.2f}% (i.e. cutoff would need to drop to {0.55*(1+lime_break_to2):.3f}, essentially to LIME's own value)")

# Range over which SHAP still beats LIME even at SHAP's worst case (score=1.0) vs LIME's best case at same shift
print("\n--- Joint worst case: how far can thresholds shift before ranking flips? ---")
for shift_pct in range(-60, 61, 1):
    shift = shift_pct / 100.0
    s = score_with_shift(coverage_bands, shap_c1, shift, 1.0)
    l = score_with_shift(r2_bands, lime_c1, shift, 1.5)
    if s <= l:
        print(f"Ranking FLIPS (SHAP <= LIME) first at shift = {shift_pct}%  (SHAP={s}, LIME={l})")
        break
else:
    print("Ranking never flips across -60% to +60% shift range")
