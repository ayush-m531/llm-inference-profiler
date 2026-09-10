"""
Tier 2: does TOKEN-POSITION structure predict per-input quality cost, when
channel-magnitude structure does not?

WHAT 11b/11c ALREADY RULED OUT: nine statistics describing the CHANNEL-wise
magnitude distribution. Tested individually (Pearson, max |r| 0.391),
non-linearly (Spearman, max 0.397) and jointly with held-out validation
(mean R2 = 0.000).

WHY POSITION IS A DIFFERENT AXIS: Sun et al. 2024 report that massive
activations sit at specific TOKEN POSITIONS - the BOS token, delimiters,
punctuation - not only in specific channels. Every statistic tested so far
collapses the token dimension (per_ch = act.abs().max(dim=0)) and therefore
cannot see positional structure at all. Different texts genuinely differ in
token structure: one long sentence versus heavy punctuation, for instance.

PRIOR, stated honestly: low. Everything measured so far says the activation's
outlier pattern is input-independent, and position is still describing the same
tensor. Estimated ~10-15% chance of a usable predictor. The realistic value is a
STRONGER NEGATIVE - having tested magnitude, non-linearity, joint fits and now
position, the limitation becomes properly closed rather than partially tested.

SCOPE: fewer layers and schemes than 11a, since the question is correlation
within cells rather than building a table.
"""
import torch
import json
import os
import math
import time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "12_position_stats"

N_TEXTS = 300
SEQ_LEN = 256
LAYERS = [3, 6, 9]                       # spread across the feasible range
CALIB_LAYER = 2
SCHEMES = {"grouped(4,4,4)": (4, 4, 4),
           "grouped(8,4,4)": (8, 4, 4),
           "grouped(8,8,4)": (8, 8, 4)}

TOP_KEEP, G2_END, G3_END = 5, 72, 1312

TEST_FRACTION = 0.3
N_SPLITS = 5

_lines = []


def out(s=""):
    print(s, flush=True)
    _lines.append(s)


def quant_group(x, n_bits):
    if x.numel() == 0 or n_bits >= 16:
        return x.clone()
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale


def apply_grouped(hidden, bits, order):
    x = hidden.float()
    top_idx = order[:TOP_KEEP]
    g2 = order[TOP_KEEP:G2_END]
    g3 = order[G2_END:G3_END]
    g4 = order[G3_END:]
    o = x.clone()
    o[..., top_idx] = x[..., top_idx]
    b2, b3, b4 = bits
    for idx, b in [(g2, b2), (g3, b3), (g4, b4)]:
        o[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
    return o


print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
H = model.config.hidden_size


def load_texts(n):
    from datasets import load_dataset
    need = SEQ_LEN + 20
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    buf, texts = "", []
    for row in ds:
        t = row["text"].strip()
        if not t or t.startswith("="):
            continue
        buf += " " + t
        if len(tok(buf)["input_ids"]) >= need:
            texts.append(buf.strip())
            buf = ""
            if len(texts) >= n:
                break
    return texts


texts = load_texts(N_TEXTS)
print(f"Loaded {len(texts)} passages\n", flush=True)


def perplexity(logits, ids):
    lp = F.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:]
    return torch.exp((-lp[torch.arange(tgt.shape[0]), tgt]).mean()).item()


def position_stats(act, ids):
    """Statistics about WHERE the magnitude sits across TOKEN POSITIONS.

    Every statistic in 11a collapsed the token dimension. These collapse the
    CHANNEL dimension instead, which is the axis Sun et al. identify as
    carrying the massive activations.

    act: [seq, hidden] float32 on GPU
    """
    a = act.abs().double()
    seq = a.shape[0]

    # magnitude carried by each TOKEN POSITION (collapse channels, not tokens)
    per_pos = a.max(dim=1).values                     # [seq] - each token's peak
    per_pos_sum = a.sum(dim=1)                        # [seq] - each token's total
    tot = per_pos.sum().item()

    srt = torch.sort(per_pos, descending=True).values
    k1 = max(1, int(round(seq * 0.01)))
    k5 = max(1, int(round(seq * 0.05)))

    # concentration across positions
    pos_top1 = srt[:k1].sum().item() / tot
    pos_top5 = srt[:k5].sum().item() / tot

    # how much does the FIRST token (BOS-like) dominate
    bos_share = per_pos[0].item() / per_pos.mean().item()

    # kurtosis of the per-position magnitudes
    v = per_pos
    m = v.mean()
    c = v - m
    var = (c ** 2).mean()
    pos_kurt = ((c ** 4).mean() / (var ** 2)).item() if var > 0 else 0.0

    # effective number of positions carrying magnitude (entropy-based)
    p = per_pos_sum / per_pos_sum.sum()
    ent = -(p * (p + 1e-12).log()).sum().item()
    eff_positions = math.exp(ent) / seq                # normalised 0..1

    # positional analogue of max_over_mean
    pos_max_over_mean = (per_pos.max() / per_pos.mean()).item()

    # do the loud CHANNELS and the loud POSITIONS coincide?
    # take the top-5 channels, see what share of THEIR magnitude sits in the
    # top-5 positions
    per_ch = a.max(dim=0).values
    top_ch = torch.topk(per_ch, 5).indices
    sub = a[:, top_ch]                                 # [seq, 5]
    sub_per_pos = sub.sum(dim=1)
    top_pos = torch.topk(sub_per_pos, 5).indices
    alignment = sub_per_pos[top_pos].sum().item() / sub_per_pos.sum().item()

    # effective rank: how much independent structure is present
    try:
        sv = torch.linalg.svdvals(act.float())
        sv = sv / sv.sum()
        sv_ent = -(sv * (sv + 1e-12).log()).sum().item()
        eff_rank = math.exp(sv_ent)
    except Exception:
        eff_rank = float("nan")

    return {"pos_top1pct": pos_top1,
            "pos_top5pct": pos_top5,
            "bos_share": bos_share,
            "pos_kurtosis": pos_kurt,
            "eff_positions": eff_positions,
            "pos_max_over_mean": pos_max_over_mean,
            "chan_pos_alignment": alignment,
            "eff_rank": eff_rank}


PREDICTORS = ["pos_top1pct", "pos_top5pct", "bos_share", "pos_kurtosis",
              "eff_positions", "pos_max_over_mean", "chan_pos_alignment",
              "eff_rank"]

records = []
t0 = time.time()
os.makedirs(OUT_DIR, exist_ok=True)

for ti, text in enumerate(texts):
    ids = tok(text, return_tensors="pt")["input_ids"][0][:SEQ_LEN]
    if ids.shape[0] < SEQ_LEN:
        continue
    ids = ids.unsqueeze(0).to("cuda")

    with torch.no_grad():
        base = model(ids, output_hidden_states=True)
    ref_probs = F.softmax(base.logits[0].float(), dim=-1)
    ref_ppl = perplexity(base.logits[0].float(), ids)

    orders, pstats = {}, {}
    for L in LAYERS:
        act = base.hidden_states[L + 1][0].float()
        orders[L] = torch.argsort(act.abs().max(dim=0).values,
                                  descending=True).to("cuda")
        pstats[L] = position_stats(act, ids)
    del base
    torch.cuda.empty_cache()

    for L in LAYERS:
        for sname, bits in SCHEMES.items():
            hidden = model.model.embed_tokens(ids)
            pos = torch.arange(SEQ_LEN).unsqueeze(0).to("cuda")
            pe = model.model.rotary_emb(hidden, pos)
            with torch.no_grad():
                for i, layer in enumerate(model.model.layers):
                    hidden = layer(hidden, attention_mask=None,
                                   position_ids=pos, position_embeddings=pe,
                                   past_key_values=None, use_cache=False)
                    if i == L:
                        hidden = apply_grouped(hidden, bits,
                                               orders[L]).to(hidden.dtype)
                hidden = model.model.norm(hidden)
                lg = model.lm_head(hidden)[0].float()
            p = F.softmax(lg, dim=-1)
            kl = F.kl_div((p + 1e-12).log(), ref_probs,
                          reduction="none").sum(-1).mean().item()

            rec = {"text_id": ti, "layer": L, "scheme": sname,
                   "kl": kl, "ref_ppl": ref_ppl}
            rec.update(pstats[L])
            records.append(rec)

    del orders
    torch.cuda.empty_cache()

    if (ti + 1) % 25 == 0:
        el = time.time() - t0
        eta = (len(texts) - ti - 1) / ((ti + 1) / el) / 60
        print(f"  {ti+1}/{len(texts)} | {el/60:.1f} min | ~{eta:.0f} min left",
              flush=True)

print(f"\nCollected {len(records):,} records in {(time.time()-t0)/60:.1f} min\n",
      flush=True)


# ---------------- analysis ----------------
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
    idx = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and xs[idx[j + 1]] == xs[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def spearman(xs, ys):
    return pearson(rankify(xs), rankify(ys))


def solve(A, b):
    n, p = len(A), len(A[0])
    ATA = [[sum(A[k][i] * A[k][j] for k in range(n)) for j in range(p)]
           for i in range(p)]
    ATb = [sum(A[k][i] * b[k] for k in range(n)) for i in range(p)]
    for i in range(p):
        ATA[i][i] += 1e-8
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


def r2(actual, pred):
    m = sum(actual) / len(actual)
    tot = sum((a - m) ** 2 for a in actual)
    res = sum((a - p) ** 2 for a, p in zip(actual, pred))
    return 1 - res / tot if tot > 0 else float("nan")


def standardise(rows):
    p = len(rows[0])
    ms, sds = [], []
    for j in range(p):
        v = [r[j] for r in rows]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        ms.append(m)
        sds.append(sd if sd > 1e-12 else 1.0)
    return [[(r[j] - ms[j]) / sds[j] for j in range(p)] for r in rows]


import random

out(f"Model: {MODEL_NAME}")
out(f"{len(records):,} records | {len(texts)} texts | seq {SEQ_LEN} "
    f"| layers {LAYERS}")
out(f"Positional predictors: {', '.join(PREDICTORS)}")
out()

out("=" * 92)
out("DO THE POSITIONAL STATISTICS EVEN VARY ACROSS TEXTS?")
out("=" * 92)
out("If they are constant, they cannot predict anything. cv = sd / mean.")
out()
out(f"{'layer':>5} " + "".join(f"{p[:13]:>15}" for p in PREDICTORS))
out("-" * 92)
for L in LAYERS:
    rows = [r for r in records if r["layer"] == L
            and r["scheme"] == "grouped(8,4,4)"]
    line = f"{L:5d} "
    for p in PREDICTORS:
        v = [r[p] for r in rows]
        m = sum(v) / len(v)
        sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
        line += f"{(sd/abs(m) if m else 0):15.4f}"
    out(line)
out()

out("=" * 92)
out("CORRELATION WITHIN A FIXED CELL  (Pearson / Spearman)")
out("=" * 92)
best_p = {p: 0.0 for p in PREDICTORS}
best_s = {p: 0.0 for p in PREDICTORS}
out(f"{'layer':>5} {'scheme':>16} " + "".join(f"{p[:13]:>15}" for p in PREDICTORS))
out("-" * 92)
for L in LAYERS:
    for s in SCHEMES:
        rows = [r for r in records if r["layer"] == L and r["scheme"] == s]
        if len(rows) < 20:
            continue
        kls = [r["kl"] for r in rows]
        line = f"{L:5d} {s:>16} "
        for p in PREDICTORS:
            xs = [r[p] for r in rows]
            c = pearson(xs, kls)
            sc = spearman(xs, kls)
            if not math.isnan(c):
                best_p[p] = max(best_p[p], abs(c))
            if not math.isnan(sc):
                best_s[p] = max(best_s[p], abs(sc))
            line += f"{c:15.3f}"
        out(line)
out()
out("Strongest |Pearson| / |Spearman| per positional predictor:")
for p in sorted(best_p, key=lambda x: -best_p[x]):
    out(f"  {p:>22}: {best_p[p]:.3f} / {best_s[p]:.3f}")
out()

out("=" * 92)
out("MULTIVARIATE, HELD-OUT VALIDATED")
out("=" * 92)
out(f"{'layer':>5} {'scheme':>16} {'in-sample R2':>14} {'held-out R2':>13} "
    f"{'implied |r|':>13}")
out("-" * 92)
cells = {}
for L in LAYERS:
    for s in SCHEMES:
        rows = [r for r in records if r["layer"] == L and r["scheme"] == s]
        if len(rows) < 50:
            continue
        X = standardise([[r[p] for p in PREDICTORS] for r in rows])
        X = [[1.0] + row for row in X]
        y = [r["kl"] for r in rows]
        coef = solve(X, y)
        if coef is None:
            continue
        r2_in = r2(y, [sum(c * xi for c, xi in zip(coef, row)) for row in X])
        outs = []
        for seed in range(N_SPLITS):
            rnd = random.Random(seed)
            idx = list(range(len(rows)))
            rnd.shuffle(idx)
            cut = int(len(idx) * (1 - TEST_FRACTION))
            c2 = solve([X[i] for i in idx[:cut]], [y[i] for i in idx[:cut]])
            if c2 is None:
                continue
            outs.append(r2([y[i] for i in idx[cut:]],
                           [sum(c * xi for c, xi in zip(c2, X[i]))
                            for i in idx[cut:]]))
        r2_out = sum(outs) / len(outs) if outs else float("nan")
        cells[f"{L}_{s}"] = {"r2_in": r2_in, "r2_out": r2_out}
        out(f"{L:5d} {s:>16} {r2_in:14.3f} {r2_out:13.3f} "
            f"{math.sqrt(max(0.0, r2_out)):13.3f}")
out()

out("=" * 92)
out("VERDICT")
out("=" * 92)
valid = [v for v in cells.values() if not math.isnan(v["r2_out"])]
bp = max(best_p.values())
bs = max(best_s.values())
br2 = max(v["r2_out"] for v in valid) if valid else 0.0
bimp = math.sqrt(max(0.0, br2))
mo = sum(v["r2_out"] for v in valid) / len(valid) if valid else 0.0
mi = sum(v["r2_in"] for v in valid) / len(valid) if valid else 0.0

out(f"  POSITIONAL predictors:")
out(f"    best single, Pearson         : {bp:.3f}")
out(f"    best single, Spearman        : {bs:.3f}")
out(f"    best combined, held-out |r|  : {bimp:.3f}")
out(f"    mean in-sample R2            : {mi:.3f}")
out(f"    mean held-out R2             : {mo:.3f}")
out()
out(f"  For comparison, CHANNEL-MAGNITUDE predictors (11b/11c):")
out(f"    best single, Pearson         : 0.391")
out(f"    best single, Spearman        : 0.397")
out(f"    best combined, held-out |r|  : 0.381")
out(f"    mean held-out R2             : 0.000")
out()
if bimp >= 0.7 or bp >= 0.7:
    out("  POSITIONAL STRUCTURE PREDICTS per-input quality cost where channel")
    out("  magnitude does not. The hardness-predictor question REOPENS.")
    out("  Next: identify which statistic carries it, validate on a second")
    out("  sequence length, then decide whether to build it.")
elif bimp >= 0.5 or bp >= 0.5:
    out("  Positional structure is a MODERATE predictor - better than channel")
    out("  magnitude but short of a quality guarantee. Report as a partial")
    out("  result; keep the p95 table for the guarantee.")
else:
    out("  Positional structure does NOT predict per-input quality cost either.")
    out()
    out("  The negative result is now comprehensive. Four independent angles:")
    out("    1. channel-magnitude statistics, individually and linearly")
    out("    2. the same, allowing monotone non-linear relationships")
    out("    3. the same, combined, with held-out validation")
    out("    4. token-POSITION structure and effective rank")
    out("  None yields a usable predictor. Per-input quality cost must be")
    out("  BUDGETED FOR (p95) rather than detected.")

with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump({"model": MODEL_NAME, "n_texts": len(texts), "seq_len": SEQ_LEN,
               "layers": LAYERS, "predictors": PREDICTORS,
               "best_pearson": best_p, "best_spearman": best_s,
               "cells": cells, "records": records}, f)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt}}")
