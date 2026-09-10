"""
Tier 1: can a COMBINATION of activation statistics predict per-input quality
cost, when no individual one could?

11b tested each statistic separately with Pearson correlation, and found a
maximum |r| of 0.391 across 480 measurements. That test has two blind spots:

  1. It tested statistics INDIVIDUALLY. Several weak predictors carrying
     partly independent information could combine into a usable one.
  2. Pearson only detects LINEAR relationships. A real but curved relationship
     would show a low r.

This closes both. No GPU - reads 11a_collect.json only.

METHOD NOTE: the multivariate fit is validated by holding out data. Fitting on
all 300 samples and reporting R2 would overstate the result, because a model
with 9 predictors can fit noise. Reported here are BOTH the in-sample R2 (what
a naive analysis would show) and the held-out R2 (what the model would actually
achieve on unseen inputs). The gap between them is itself informative.
"""
import json
import os
import math
import random

IN_PATH = "/home/ayush.thakar/thesis/experiments_llama/results/11a_collect.json"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "11c_multivariate"

# max_abs is excluded: it is constant across texts (outliers are injected once
# and ride the residual unchanged), so it produced nan in every correlation.
PREDICTORS = ["kurtosis", "skew", "top1pct", "top5pct",
              "top5_channels_share", "mean_abs", "max_over_mean", "std"]

SEQ = 256
TEST_FRACTION = 0.3
N_SPLITS = 5          # repeat the train/test split this many times

_lines = []


def out(s=""):
    print(s)
    _lines.append(s)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def rankify(xs):
    """Convert values to ranks, averaging ties."""
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[idx[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Rank correlation - detects any MONOTONE relationship, not just linear."""
    return pearson(rankify(xs), rankify(ys))


def solve(A, b):
    """Least squares via normal equations with Gaussian elimination.
    A is n x p, b is length n. Returns p coefficients, or None if singular."""
    n, p = len(A), len(A[0])
    # normal equations: (A^T A) x = A^T b
    ATA = [[sum(A[k][i] * A[k][j] for k in range(n)) for j in range(p)]
           for i in range(p)]
    ATb = [sum(A[k][i] * b[k] for k in range(n)) for i in range(p)]

    # small ridge term for numerical stability
    for i in range(p):
        ATA[i][i] += 1e-8

    # gaussian elimination with partial pivoting
    M = [row[:] + [ATb[i]] for i, row in enumerate(ATA)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            return None
        M[col], M[piv] = M[piv], M[col]
        for r in range(p):
            if r == col:
                continue
            f = M[r][col] / M[col][col]
            for c in range(col, p + 1):
                M[r][c] -= f * M[col][c]
    return [M[i][p] / M[i][i] for i in range(p)]


def r_squared(actual, predicted):
    m = sum(actual) / len(actual)
    ss_tot = sum((a - m) ** 2 for a in actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def standardise(cols):
    """Zero mean, unit variance per column. Keeps the fit numerically sane
    when predictors differ by orders of magnitude (kurtosis ~130000 vs
    top1pct ~0.6)."""
    p = len(cols[0])
    means, sds = [], []
    for j in range(p):
        vals = [r[j] for r in cols]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))
        means.append(m)
        sds.append(sd if sd > 1e-12 else 1.0)
    return [[(r[j] - means[j]) / sds[j] for j in range(p)] for r in cols]


print("Loading 11a_collect.json ...")
data = json.load(open(IN_PATH))
R = [r for r in data["records"]
     if r["seq_len"] == SEQ and r["order"] == "per_layer"]
LAYERS = data["layers"]
SCHEMES = [s for s in ["grouped(4,4,4)", "grouped(8,4,4)",
                       "grouped(8,8,4)", "grouped(8,8,8)"]
           if s in data["schemes"]]

out(f"Records at seq_len {SEQ}, per-layer ordering: {len(R):,}")
out(f"Predictors tested together: {len(PREDICTORS)}")
out(f"  {', '.join(PREDICTORS)}")
out(f"(max_abs excluded - constant across texts, produced nan in 11b)")
out(f"Held-out fraction {TEST_FRACTION}, averaged over {N_SPLITS} random splits")
out()

# ============================================================
out("=" * 92)
out("A. SPEARMAN RANK CORRELATION  (catches monotone non-linear relationships)")
out("=" * 92)
out("11b used Pearson, which only sees straight lines. If KL depends on a")
out("statistic through a curve, Pearson would miss it and Spearman would not.")
out()
out(f"{'layer':>5} {'scheme':>16} " + "".join(f"{p[:11]:>13}" for p in PREDICTORS))
out("-" * 92)
best_spear = {p: 0.0 for p in PREDICTORS}
for L in LAYERS:
    for s in SCHEMES:
        rows = [r for r in R if r["layer"] == L and r["scheme"] == s]
        if len(rows) < 20:
            continue
        kls = [r["kl"] for r in rows]
        line = f"{L:5d} {s:>16} "
        for p in PREDICTORS:
            c = spearman([r[p] for r in rows], kls)
            if not math.isnan(c):
                best_spear[p] = max(best_spear[p], abs(c))
            line += f"{c:13.3f}"
        out(line)
out()
out("Strongest |Spearman| per predictor, across all cells:")
for p in sorted(best_spear, key=lambda x: -best_spear[x]):
    out(f"  {p:>22}: {best_spear[p]:.3f}")
out()

# ============================================================
out("=" * 92)
out("B. MULTIVARIATE FIT  (all predictors together)")
out("=" * 92)
out("in-sample R2  = fit and evaluated on the same data. OPTIMISTIC - with 8")
out("                predictors and 300 points it can fit noise.")
out("held-out R2   = fit on 70%, evaluated on the unseen 30%. This is what the")
out("                model would actually achieve on new inputs.")
out("A large gap between them means the model is memorising, not learning.")
out()
out(f"{'layer':>5} {'scheme':>16} {'n':>5} {'in-sample R2':>14} "
    f"{'held-out R2':>13} {'implied |r|':>13} {'best single':>13}")
out("-" * 92)

results = {}
best_heldout = -1.0
best_cell = None

for L in LAYERS:
    for s in SCHEMES:
        rows = [r for r in R if r["layer"] == L and r["scheme"] == s]
        if len(rows) < 50:
            continue
        X = standardise([[r[p] for p in PREDICTORS] for r in rows])
        X = [[1.0] + row for row in X]          # intercept
        y = [r["kl"] for r in rows]

        # in-sample
        coef = solve(X, y)
        if coef is None:
            continue
        pred = [sum(c * xi for c, xi in zip(coef, row)) for row in X]
        r2_in = r_squared(y, pred)

        # held out, averaged over several random splits
        r2s = []
        for seed in range(N_SPLITS):
            rnd = random.Random(seed)
            idx = list(range(len(rows)))
            rnd.shuffle(idx)
            cut = int(len(idx) * (1 - TEST_FRACTION))
            tr, te = idx[:cut], idx[cut:]
            c2 = solve([X[i] for i in tr], [y[i] for i in tr])
            if c2 is None:
                continue
            pt = [sum(c * xi for c, xi in zip(c2, X[i])) for i in te]
            r2s.append(r_squared([y[i] for i in te], pt))
        r2_out = sum(r2s) / len(r2s) if r2s else float("nan")

        # strongest single predictor in this cell, for comparison
        singles = [abs(pearson([r[p] for r in rows], y)) for p in PREDICTORS]
        singles = [x for x in singles if not math.isnan(x)]
        best_single = max(singles) if singles else 0.0

        implied = math.sqrt(max(0.0, r2_out))
        results[f"{L}_{s}"] = {"n": len(rows), "r2_in": r2_in,
                               "r2_out": r2_out, "implied_r": implied,
                               "best_single_r": best_single}
        if r2_out > best_heldout:
            best_heldout, best_cell = r2_out, f"L{L} {s}"

        out(f"{L:5d} {s:>16} {len(rows):5d} {r2_in:14.3f} {r2_out:13.3f} "
            f"{implied:13.3f} {best_single:13.3f}")

out()

# ============================================================
out("=" * 92)
out("VERDICT")
out("=" * 92)

valid = [v for v in results.values() if not math.isnan(v["r2_out"])]
if valid:
    best_r2 = max(v["r2_out"] for v in valid)
    best_imp = max(v["implied_r"] for v in valid)
    mean_in = sum(v["r2_in"] for v in valid) / len(valid)
    mean_out = sum(v["r2_out"] for v in valid) / len(valid)
    best_single_overall = max(v["best_single_r"] for v in valid)
    best_spear_overall = max(best_spear.values())

    out(f"  best held-out R2 across cells : {best_r2:.3f}  ({best_cell})")
    out(f"  implied |r| for that cell     : {best_imp:.3f}")
    out(f"  mean in-sample R2             : {mean_in:.3f}")
    out(f"  mean held-out R2              : {mean_out:.3f}")
    out(f"  overfitting gap               : {mean_in - mean_out:.3f}")
    out()
    out(f"  best SINGLE predictor (Pearson, 11b) : {best_single_overall:.3f}")
    out(f"  best SINGLE predictor (Spearman)     : {best_spear_overall:.3f}")
    out(f"  best COMBINED (held-out, implied |r|): {best_imp:.3f}")
    out()

    if best_imp >= 0.7:
        out("  A COMBINATION of statistics predicts per-input quality cost well")
        out("  enough to be usable. The hardness-predictor question REOPENS.")
        out("  Next: check which predictors carry the weight, verify the fit")
        out("  holds on a second sequence length, then build it.")
    elif best_imp >= 0.5:
        out("  Moderate combined predictive power - better than any single")
        out("  statistic, but not enough for a quality guarantee. Could serve as")
        out("  a rough signal alongside the p95 table, not as a replacement.")
    else:
        out("  Combining the statistics does NOT rescue prediction. Neither does")
        out("  allowing non-linear (rank) relationships. The conclusion from 11b")
        out("  stands and is now stronger:")
        out()
        out("    No combination of these activation statistics, linear or")
        out("    monotone, predicts per-input quality cost within a fixed layer")
        out("    and scheme. Per-input variation must be BUDGETED FOR (the p95")
        out("    table) rather than detected.")
        out()
        out("  Why this is expected: the predictors are heavily redundant. skew")
        out("  mirrors kurtosis; top1pct, top5pct and top5_channels_share")
        out("  measure the same concentration at different cutoffs. Combining")
        out("  redundant weak predictors adds little.")

    if mean_in - mean_out > 0.15:
        out()
        out(f"  NOTE: the in-sample R2 averages {mean_in:.3f} while held-out")
        out(f"  averages {mean_out:.3f}. Reporting the in-sample figure would")
        out("  have overstated the result substantially - the model is largely")
        out("  fitting noise. This is why the held-out evaluation matters.")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump({"seq_len": SEQ, "predictors": PREDICTORS,
               "test_fraction": TEST_FRACTION, "n_splits": N_SPLITS,
               "best_spearman": best_spear, "cells": results}, f, indent=2)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt}}")
