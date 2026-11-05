"""
15_ppl_check: does the perplexity claim hold across 300 texts?

WHY: an earlier write-up said the compression costs "at most 3.01%"
perplexity. That figure came from 2 calibration texts (06, 06_01). But 11a
already saved a perplexity rise for every one of its 86,400 records, so the
claim can be checked at scale with NO GPU - this script only reads JSON.

WHAT IT PRINTS
  A. The configurations the controller actually chose in the socket demo:
     mean / median / p95 / max perplexity rise, and the share of texts over
     1% and over 3.01%, at each sequence length.
  B. p95 perplexity rise for every layer x scheme (seq 256).
  C. For records that MEET each KL budget: how large can the perplexity rise
     still be? This tests whether "strict KL" really means "small damage".
  D. The noise floor: how often the rise is negative. On one passage a small
     negative rise can happen by chance (rounding happens to help a few
     tokens), so the size of the negatives shows how big plain noise is.

No automated verdict: read the numbers, then decide the wording.

CAVEAT IN THE DATA: 11a built each per-layer ordering from the SAME text it
then measured (an oracle ordering). Exp 08 showed orderings from unrelated
text pick the same scheme in 88-94% of cells, so a deployed edge using a
shipped table may do slightly worse than these numbers.
"""
import json
import os
import numpy as np

IN_PATH = "/home/ayush.thakar/thesis/experiments_llama/results/11a_collect.json"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "15_ppl_check"

# (label, layer, scheme) the controller chose in the socket demo
PICKS = [("strict, 7.81 GB free", 6, "grouped(8,4,4)"),
         ("strict, 5.90 GB free", 1, "grouped(8,8,4)"),
         ("relaxed, 7.81 GB free", 8, "grouped(4,4,4)"),
         ("early stop at L3", 3, "grouped(8,8,4)")]
KL_BUDGETS = [("strict", 0.05), ("balanced", 0.10), ("relaxed", 0.25)]
TABLE_SEQ = 256

_lines = []


def out(s=""):
    print(s, flush=True)
    _lines.append(s)


def summ(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return None
    return {"n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "max": float(v.max()),
            "share_over_1pct": float((v > 1.0).mean()),
            "share_over_3_01pct": float((v > 3.01).mean()),
            "share_negative": float((v < 0).mean())}


print("Loading 11a_collect.json ...", flush=True)
data = json.load(open(IN_PATH))
R = [r for r in data["records"] if r["order"] == "per_layer"]
SEQS = sorted(set(r["seq_len"] for r in R))
LAYERS = sorted(set(r["layer"] for r in R))
SCHEMES = [s for s in ["grouped(4,4,4)", "grouped(8,4,4)",
                       "grouped(8,8,4)", "grouped(8,8,8)"]
           if s in data["schemes"]]

ref = {}
for r in R:
    ref[(r["text_id"], r["seq_len"])] = r["ref_ppl"]
refs = np.array(list(ref.values()), float)
out(f"Records (per-layer ordering): {len(R):,}")
out(f"Texts: {len(set(r['text_id'] for r in R))}   lengths: {SEQS}")
out(f"Reference perplexity: median {np.median(refs):.2f}, "
    f"min {refs.min():.2f}, max {refs.max():.2f}")
out()

result = {"picks": {}, "table_p95": {}, "kl_budget": {}, "noise": {}}

# ---------- A ----------
out("=" * 92)
out("A. THE CONTROLLER'S ACTUAL PICKS - perplexity rise in %, across texts")
out("=" * 92)
out(f"{'configuration':>24} {'seq':>5} {'n':>4} {'mean':>7} {'median':>7} "
    f"{'p95':>7} {'max':>8} {'>1%':>6} {'>3.01%':>7}")
out("-" * 92)
for label, L, s in PICKS:
    for q in SEQS:
        v = [r["ppl_rise_pct"] for r in R
             if r["layer"] == L and r["scheme"] == s and r["seq_len"] == q]
        st = summ(v)
        if st is None:
            continue
        result["picks"][f"{label} | L{L} {s} | seq {q}"] = st
        out(f"{label:>24} {q:5d} {st['n']:4d} {st['mean']:7.2f} "
            f"{st['median']:7.2f} {st['p95']:7.2f} {st['max']:8.2f} "
            f"{100*st['share_over_1pct']:5.1f}% "
            f"{100*st['share_over_3_01pct']:6.1f}%")
    out()

# ---------- B ----------
out("=" * 92)
out(f"B. p95 PERPLEXITY RISE (%) BY LAYER AND SCHEME, seq {TABLE_SEQ}")
out("=" * 92)
out(f"{'layer':>5}" + "".join(f"{s:>18}" for s in SCHEMES))
out("-" * 92)
for L in LAYERS:
    row = f"{L:5d}"
    for s in SCHEMES:
        v = [r["ppl_rise_pct"] for r in R
             if r["layer"] == L and r["scheme"] == s
             and r["seq_len"] == TABLE_SEQ]
        st = summ(v)
        if st is None:
            row += f"{'-':>18}"
            continue
        result["table_p95"][f"L{L} {s}"] = st["p95"]
        row += f"{st['p95']:18.2f}"
    out(row)
out()

# ---------- C ----------
out("=" * 92)
out(f"C. RECORDS THAT MEET EACH KL BUDGET "
    f"(layers 1-11, all schemes, seq {TABLE_SEQ})")
out("=" * 92)
out("If a KL budget really means small damage, the perplexity rise of records")
out("inside the budget should stay small.")
out()
out(f"{'budget':>10} {'KL <=':>7} {'n':>6} {'median':>8} {'p95':>8} "
    f"{'max':>9} {'>1%':>7} {'>3.01%':>8}")
out("-" * 92)
for name, b in KL_BUDGETS:
    v = [r["ppl_rise_pct"] for r in R
         if r["layer"] >= 1 and r["seq_len"] == TABLE_SEQ and r["kl"] <= b]
    st = summ(v)
    if st is None:
        continue
    result["kl_budget"][name] = st
    out(f"{name:>10} {b:7.2f} {st['n']:6d} {st['median']:8.2f} "
        f"{st['p95']:8.2f} {st['max']:9.2f} "
        f"{100*st['share_over_1pct']:6.1f}% "
        f"{100*st['share_over_3_01pct']:7.1f}%")
out()

# ---------- D ----------
out("=" * 92)
out("D. NOISE FLOOR - how far below zero single-passage rises go")
out("=" * 92)
allv = np.array([r["ppl_rise_pct"] for r in R], float)
neg = allv[allv < 0]
result["noise"] = {
    "share_negative": float((allv < 0).mean()),
    "most_negative": float(neg.min()) if neg.size else 0.0,
    "p5_of_negatives": float(np.percentile(neg, 5)) if neg.size else 0.0}
out(f"  share of all records with a negative rise : "
    f"{100*(allv < 0).mean():.1f}%")
if neg.size:
    out(f"  most negative                             : {neg.min():.2f}%")
    out(f"  5th percentile of the negatives           : "
        f"{np.percentile(neg, 5):.2f}%")
out()
out("No automated verdict: paste this output and decide the wording from it.")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump(result, f, indent=2)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt}}")
