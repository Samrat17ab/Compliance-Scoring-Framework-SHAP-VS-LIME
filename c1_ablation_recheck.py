#!/usr/bin/env python3
"""Independent re-verification of the C1 threshold ablation, using a
different code path than the original script (inline if/elif blocks
copied verbatim from the paper's actual score_coverage/score_fidelity
functions, rather than the generic table-lookup helper), plus an
independent-shift test (coverage and R^2 shifted separately, not
jointly) to check the joint-shift framing isn't hiding anything."""

shap_c1 = 0.700
lime_c1 = 0.251

def score_coverage(v, shift=0.0):
    # verbatim from csf_evaluation.py, cutoffs multiplied by (1+shift)
    if v >= 0.90 * (1 + shift): return 5.0
    if v >= 0.80 * (1 + shift): return 4.0
    if v >= 0.70 * (1 + shift): return 3.0
    if v >= 0.60 * (1 + shift): return 2.0
    return 1.0

def score_fidelity(v, shift=0.0):
    if v >= 0.85 * (1 + shift): return 5.0
    if v >= 0.75 * (1 + shift): return 4.0
    if v >= 0.65 * (1 + shift): return 3.0
    if v >= 0.55 * (1 + shift): return 2.0
    return 1.5

# --- Sanity check: shift=0 reproduces the paper's published scores exactly ---
assert score_coverage(shap_c1, 0.0) == 3.0, "MISMATCH: shift=0 should give 3.0"
assert score_fidelity(lime_c1, 0.0) == 1.5, "MISMATCH: shift=0 should give 1.5"
print("Sanity check passed: shift=0 reproduces SHAP=3.0, LIME=1.5 (matches Table V).")

# --- Cross-check 1: joint shift, independent re-implementation ---
print("\n--- Joint shift (both tables shifted together), re-implemented independently ---")
flip_shift = None
for pct in range(-60, 61, 1):
    shift = pct / 100.0
    s = score_coverage(shap_c1, shift)
    l = score_fidelity(lime_c1, shift)
    if s <= l and flip_shift is None:
        flip_shift = pct
print(f"Joint-shift ranking flips at +{flip_shift}% (independent re-implementation)")

# --- Cross-check 2: shift ONLY the coverage table, hold R^2 fixed ---
print("\n--- Coverage-only shift (R^2 table held at original thresholds) ---")
flip_shift_cov_only = None
for pct in range(-60, 61, 1):
    shift = pct / 100.0
    s = score_coverage(shap_c1, shift)
    l = score_fidelity(lime_c1, 0.0)  # unshifted
    if s <= l and flip_shift_cov_only is None:
        flip_shift_cov_only = pct
print(f"Coverage-only shift: ranking flips at +{flip_shift_cov_only}%")

# --- Cross-check 3: shift ONLY the R^2 table, hold coverage fixed ---
print("\n--- R^2-only shift (Coverage table held at original thresholds) ---")
flip_shift_r2_only = None
for pct in range(-95, 500, 1):  # need a much wider range since LIME is so far below
    shift = pct / 100.0
    s = score_coverage(shap_c1, 0.0)  # unshifted, stays 3.0
    l = score_fidelity(lime_c1, shift)
    if s <= l and flip_shift_r2_only is None:
        flip_shift_r2_only = pct
        break
print(f"R^2-only shift: ranking flips at shift = {flip_shift_r2_only}% "
      f"(i.e. R^2 thresholds would need to be lowered this much)")

# --- Cross-check 4: exact algebraic break-even points (no loop/rounding) ---
print("\n--- Exact algebraic break-even points ---")
print(f"SHAP coverage 3.0->2.0 break-even: shift = {(shap_c1/0.70 - 1)*100:.4f}%")
print(f"SHAP coverage 2.0->1.0 break-even: shift = {(shap_c1/0.60 - 1)*100:.4f}%")
print(f"LIME R^2 1.5->2.0 break-even:      shift = {(lime_c1/0.55 - 1)*100:.4f}%")

# --- Cross-check 5: verify monotonicity assumption holds (no band-order inversion) ---
print("\n--- Monotonicity check: do cutoffs stay in descending order across tested range? ---")
bad = []
for pct in range(-95, 500, 1):
    shift = pct/100.0
    cuts = [0.90*(1+shift), 0.80*(1+shift), 0.70*(1+shift), 0.60*(1+shift)]
    if cuts != sorted(cuts, reverse=True):
        bad.append(pct)
print(f"Order violations found at: {bad if bad else 'none (order preserved throughout)'}")
