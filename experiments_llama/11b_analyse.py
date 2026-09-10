"""
Analysis of the bulk collection (11a_collect.json).

Answers four questions:

  1. PERCENTILE LOOKUP TABLE. The controller's table was calibrated on ONE
     text. What does the distribution across 300 texts actually look like, and
     what should the table say if it is to meet a quality budget for most
     inputs rather than for one lucky sample?

  2. CAN ANY CHEAP STATISTIC PREDICT PER-INPUT QUALITY COST? Measured WITHIN a
     fixed (layer, scheme, seq_len). Pooling across layers would recover the
     already-known layer effect and say nothing useful.

  3. DOES SEQUENCE LENGTH MATTER? All prior quality work used one length. The
     old table said L3/grouped(8,4,4) gives KL 0.0221 at seq 100; the bulk data
     shows mean 0.083 at seq 256. How much of that gap is length?

  4. FROZEN vs PER-LAYER ORDERING across 300 texts, not 2.

Reads only. No GPU.
"""
import json
import os
import math
import statistics as st

IN_PATH = "/home/ayush.thakar/thesis/experiments_llama/results/11a_collect.json"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "11b_analyse"

PREDICTORS = ["kurtosis", "skew", "top1pct", "top5pct",
              "top5_channels_share", "max_abs", "mean_abs",
              "max_over_mean", "std", "ref_ppl"]

KL_BUDGETS = [0.05, 0.10, 0.25]

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


def pct(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[i]


print("Loading 11a_collect.json ...")
data = json.load(open(IN_PATH))
R = data["records"]
LAYERS = data["layers"]
SCHEMES = data["schemes"]
SEQ_LENS = data["seq_lens"]

out(f"Source data: {len(R):,} records from {data['n_texts']} texts")
out(f"Text source: {data['source']}")
out(f"Layers {LAYERS}  |  seq_lens {SEQ_LENS}  |  schemes {SCHEMES}")
out()


def sel(**kw):
    return [r for r in R if all(r[k] == v for k, v in kw.items())]


# ============================================================
out("=" * 90)
out("1. PERCENTILE LOOKUP TABLE  (per-layer ordering, the deployed design)")
out("=" * 90)
out("The controller's hardcoded table came from ONE text. These are the actual")
out("distributions. A table built on the MEAN meets the budget for about half")
out("of inputs; a table built on p95 meets it for 95%.")
out()

table = {}
for sl in SEQ_LENS:
    out(f"--- seq_len {sl} ---")
    out(f"{'layer':>5} {'scheme':>16} {'mean':>9} {'sd':>9} {'p50':>9} "
        f"{'p95':>9} {'max':>9} {'p95/mean':>9}")
    out("-" * 90)
    for L in LAYERS:
        for s in SCHEMES:
            rows = sel(layer=L, scheme=s, seq_len=sl, order="per_layer")
            if not rows:
                continue
            ks = sorted(r["kl"] for r in rows)
            m = sum(ks) / len(ks)
            sd = st.pstdev(ks)
            p50, p95 = pct(ks, 0.50), pct(ks, 0.95)
            table[f"{sl}_{L}_{s}"] = {"mean": m, "sd": sd, "p50": p50,
                                      "p95": p95, "max": ks[-1],
                                      "min": ks[0], "n": len(ks)}
            out(f"{L:5d} {s:>16} {m:9.5f} {sd:9.5f} {p50:9.5f} "
                f"{p95:9.5f} {ks[-1]:9.5f} {p95/m if m else 0:9.2f}")
    out()

# ============================================================
out("=" * 90)
out("2. WHAT THE TABLE CHANGE COSTS: scheme picked on MEAN vs on p95")
out("=" * 90)
out("A controller selecting on the mean will violate the budget for roughly")
out("half its inputs. Selecting on p95 is honest but may force a dearer scheme.")
out()

SCHEME_ORDER = [s for s in ["grouped(4,4,4)", "grouped(8,4,4)",
                            "grouped(8,8,4)", "grouped(8,8,8)"]
                if s in SCHEMES]


def pick(sl, L, budget, stat):
    for s in SCHEME_ORDER:
        k = f"{sl}_{L}_{s}"
        if k in table and table[k][stat] <= budget:
            return s
    return "none passes"


for sl in SEQ_LENS:
    out(f"--- seq_len {sl} ---")
    out(f"{'layer':>5} " + "".join(
        f"{'KL<=' + str(b):>34}" for b in KL_BUDGETS))
    out(f"{'':5} " + "".join(f"{'on mean':>17}{'on p95':>17}"
                             for _ in KL_BUDGETS))
    out("-" * 90)
    for L in LAYERS:
        line = f"{L:5d} "
        for b in KL_BUDGETS:
            line += f"{pick(sl, L, b, 'mean'):>17}{pick(sl, L, b, 'p95'):>17}"
        out(line)
    out()

# ============================================================
out("=" * 90)
out("3. SEQUENCE LENGTH EFFECT")
out("=" * 90)
out("Mean KL at each length, per-layer ordering. If KL grows with length, the")
out("lookup table needs a length dimension - it currently has none.")
out()
for s in SCHEME_ORDER:
    out(f"--- {s} ---")
    out(f"{'layer':>5}" + "".join(f"{'seq '+str(sl):>14}" for sl in SEQ_LENS)
        + f"{'512/128':>12}")
    out("-" * 70)
    for L in LAYERS:
        vals = []
        line = f"{L:5d}"
        for sl in SEQ_LENS:
            k = f"{sl}_{L}_{s}"
            v = table[k]["mean"] if k in table else float("nan")
            vals.append(v)
            line += f"{v:14.5f}"
        ratio = vals[-1] / vals[0] if vals[0] else float("nan")
        out(line + f"{ratio:12.2f}")
    out()

# ============================================================
out("=" * 90)
out("4. THE PREDICTOR QUESTION: correlation WITHIN a fixed cell")
out("=" * 90)
out("Each row is one (seq_len, layer, scheme) cell with n=300 texts. A strong")
out("|r| here means a cheap statistic could predict per-input quality cost at")
out("runtime. Pooled correlation is reported afterwards for contrast - it will")
out("look better and mean less.")
out()

within = {}
for sl in [256]:                       # representative; full set is in JSON
    for L in LAYERS:
        for s in SCHEME_ORDER:
            rows = sel(layer=L, scheme=s, seq_len=sl, order="per_layer")
            if len(rows) < 10:
                continue
            kls = [r["kl"] for r in rows]
            for p in PREDICTORS:
                within[(sl, L, s, p)] = pearson([r[p] for r in rows], kls)

out(f"seq_len 256, per-layer ordering")
out(f"{'layer':>5} {'scheme':>16} " + "".join(f"{p[:11]:>12}"
                                              for p in PREDICTORS))
out("-" * 90)
for L in LAYERS:
    for s in SCHEME_ORDER:
        if (256, L, s, PREDICTORS[0]) not in within:
            continue
        line = f"{L:5d} {s:>16} "
        for p in PREDICTORS:
            line += f"{within[(256, L, s, p)]:12.3f}"
        out(line)
out()

out("Strongest |r| achieved by each predictor, across all cells:")
best = {}
for p in PREDICTORS:
    vals = [abs(v) for (sl, L, s, pp), v in within.items()
            if pp == p and not math.isnan(v)]
    best[p] = max(vals) if vals else 0.0
for p in sorted(best, key=lambda x: -best[x]):
    out(f"  {p:>22}: {best[p]:.3f}")
out()

out("Pooled across all layers (the misleading comparison):")
for p in PREDICTORS:
    rows = sel(seq_len=256, order="per_layer")
    c = pearson([r[p] for r in rows], [r["kl"] for r in rows])
    out(f"  {p:>22}: {c:6.3f}")

# ============================================================
out()
out("=" * 90)
out("5. FROZEN vs PER-LAYER ORDERING  (300 texts, previously 2)")
out("=" * 90)
out(f"{'seq':>5} {'layer':>5} {'scheme':>16} {'mean KL froz':>14} "
    f"{'mean KL own':>13} {'ratio':>8}")
out("-" * 90)
ord_summary = {}
for sl in SEQ_LENS:
    for L in LAYERS:
        for s in SCHEME_ORDER:
            fr = sel(layer=L, scheme=s, seq_len=sl, order="frozen")
            pl = sel(layer=L, scheme=s, seq_len=sl, order="per_layer")
            if not fr or not pl:
                continue
            mf = sum(r["kl"] for r in fr) / len(fr)
            mp = sum(r["kl"] for r in pl) / len(pl)
            ord_summary[f"{sl}_{L}_{s}"] = {"frozen": mf, "per_layer": mp,
                                            "ratio": mf / mp if mp else 0}
            if sl == 256:
                out(f"{sl:5d} {L:5d} {s:>16} {mf:14.5f} {mp:13.5f} "
                    f"{mf/mp if mp else 0:8.2f}")
out()

# ============================================================
out("=" * 90)
out("VERDICT")
out("=" * 90)

strongest = max(best.values()) if best else 0.0
top_pred = max(best, key=lambda x: best[x]) if best else "none"
out(f"Best within-cell predictor: {top_pred}, |r| = {strongest:.3f}")
out()
if strongest >= 0.7:
    out("  A cheap statistic DOES predict per-input quality cost within a fixed")
    out("  layer and scheme. A runtime hardness predictor is viable. Next step:")
    out("  fit it, hold out a test set, and check the fit generalises.")
elif strongest >= 0.4:
    out("  Moderate within-cell correlation. Enough for a rough signal, not")
    out("  enough for a quality guarantee. Report as a partial result and use")
    out("  the p95 table for the actual guarantee.")
else:
    out("  NO usable within-cell correlation. Quality cost varies substantially")
    out("  across inputs, but none of these cheap statistics tracks it. A")
    out("  runtime hardness predictor built on them is not viable.")
    out("  => the p95 lookup table is the correct answer: it does not need to")
    out("     detect hard inputs, it simply budgets for them.")
out()
if "ref_ppl" in best and best["ref_ppl"] > 0.5 and best["ref_ppl"] >= strongest - 0.1:
    out("  NOTE: reference perplexity is among the strongest predictors. That is")
    out("  informative but NOT deployable - computing it requires running the")
    out("  uncompressed model and knowing the true next tokens, neither of which")
    out("  the edge has. So text difficulty drives the variation, but difficulty")
    out("  is not observable from the activation alone.")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump({"n_records": len(R), "n_texts": data["n_texts"],
               "source": data["source"],
               "percentile_table": table,
               "ordering_comparison": ord_summary,
               "within_cell_r": {f"{sl}_{L}_{s}_{p}": v
                                 for (sl, L, s, p), v in within.items()},
               "best_within_cell_r": best}, f, indent=2)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt}}")
