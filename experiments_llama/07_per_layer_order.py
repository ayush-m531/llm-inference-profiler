"""
Per-layer channel ordering vs a single frozen order.

Current design: one channel order, frozen from layer 2, reused at every split.
Proposed design: a table of orders, one per layer, precomputed offline and
shipped to both edge and cloud. At runtime the edge sends only the integer k
(which layer it cut at) - already required by the protocol - so per-layer
ordering costs zero extra runtime bytes.

Storage cost: 12 layers x 4096 channels x 2 bytes = ~98 KB against a 14 GB
model. Negligible.

Measures, for layers 0-11 (the memory-feasible split range):
  1. How much the ordering drifts from layer 2's, by GROUP membership
     (group assignment is what the compression actually cares about, not rank)
  2. Whether per-layer ordering improves KL and perplexity
  3. THE DECIDER: whether it changes which SCHEME gets picked at each budget

Only (3) moves bytes. Schemes come in 7 discrete steps, so a quality
improvement only saves bytes if it crosses a threshold.

Writes JSON and a TXT transcript.
"""
import torch
import json
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "07_per_layer_order"

# feasible split range from 04_memory_ceiling (L7 busy, L11 cleared at bf16)
SPLITS = list(range(0, 12))

TOP_KEEP = 5
G2_END = 72
G3_END = 1312
CALIB_LAYER = 2          # the layer the single frozen order comes from

PPL_SEQ_LEN = 256
PPL_BUDGETS = [1.0, 3.0, 5.0, 10.0]

PPL_TEXT = ("Climate scientists have documented a steady rise in global average "
            "temperatures over the past century, driven primarily by the "
            "accumulation of greenhouse gases in the atmosphere. The consequences "
            "extend well beyond simple warming: shifting rainfall patterns, more "
            "frequent extreme weather events, rising sea levels, and disruptions "
            "to agriculture and ecosystems. Mitigation strategies focus on "
            "reducing emissions through renewable energy adoption, improved "
            "efficiency, and changes in land use, while adaptation measures aim "
            "to reduce vulnerability to changes already underway. The pace of "
            "both remains a subject of intense policy debate across nations, "
            "with developing countries arguing that historical responsibility "
            "should shape the distribution of costs and obligations. ") * 3


_lines = []


def out(s=""):
    print(s)
    _lines.append(s)


def quant_group(x, n_bits):
    if x.numel() == 0 or n_bits >= 16:
        return x.clone()
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def apply_grouped(hidden, bits_combo, order):
    x = hidden.float()
    seq_len = x.shape[1]
    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:G2_END]
    g3_idx = order[G2_END:G3_END]
    g4_idx = order[G3_END:]
    o = x.clone()
    o[..., top_idx] = x[..., top_idx]
    b2, b3, b4 = bits_combo
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        o[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
    nbytes = (TOP_KEEP * seq_len * 16) / 8
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        nbytes += (len(idx) * seq_len * b) / 8 + 4
    return o, nbytes


SCHEMES = [
    ("grouped", (4, 4, 4)),
    ("grouped", (8, 4, 4)),
    ("grouped", (8, 8, 4)),
    ("grouped", (8, 8, 8)),
]

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
H = model.config.hidden_size

ids = tok(PPL_TEXT, return_tensors="pt")["input_ids"][0][:PPL_SEQ_LEN].unsqueeze(0).to("cuda")


def forward_with_scheme(split, scheme, order):
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    nbytes = None
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None,
                       use_cache=False)
        if split is not None and i == split:
            od = hidden.dtype
            q, nbytes = apply_grouped(hidden, scheme[1], order)
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0].float(), nbytes


def perplexity(logits):
    lp = F.log_softmax(logits[:-1], dim=-1)
    targets = ids[0, 1:]
    nll = -lp[torch.arange(targets.shape[0]), targets]
    return torch.exp(nll.mean()).item()


# ---- build one order per layer from a single calibration pass ----
print("Building per-layer channel orders...")
with torch.no_grad():
    hs = model(ids, output_hidden_states=True).hidden_states

orders = {}
for L in range(n_layers):
    act = hs[L + 1][0].float()
    orders[L] = torch.argsort(act.abs().max(dim=0).values,
                              descending=True).to("cuda")
del hs

FROZEN = orders[CALIB_LAYER]

with torch.no_grad():
    ref_logits, _ = forward_with_scheme(None, None, FROZEN)
    ref_probs = F.softmax(ref_logits, dim=-1)
    ref_ppl = perplexity(ref_logits)

out(f"Model: {MODEL_NAME}")
out(f"Feasible split range tested: L{SPLITS[0]}-L{SPLITS[-1]}")
out(f"Single frozen order taken from layer {CALIB_LAYER}")
out(f"Reference (uncompressed) perplexity: {ref_ppl:.4f}")
out()


def group_of(order):
    """Map channel index -> group id (0=top, 1=shoulder, 2=mid, 3=bulk)."""
    g = torch.empty(H, dtype=torch.long, device=order.device)
    g[order[:TOP_KEEP]] = 0
    g[order[TOP_KEEP:G2_END]] = 1
    g[order[G2_END:G3_END]] = 2
    g[order[G3_END:]] = 3
    return g


out("=" * 78)
out("ORDERING DRIFT FROM THE FROZEN ORDER")
out("=" * 78)
out("Fraction of channels assigned to the SAME group as under layer "
    f"{CALIB_LAYER}'s order.")
out("Group membership is what compression acts on; exact rank within a group")
out("does not matter.")
out()
out(f"{'layer':>5} {'same group':>12} {'top5 kept':>11} {'top72 kept':>12} "
    f"{'loud missed':>13}")
out("-" * 60)

frozen_groups = group_of(FROZEN)
drift = {}
for L in SPLITS:
    g = group_of(orders[L])
    same = (g == frozen_groups).float().mean().item()
    top5_own = set(orders[L][:TOP_KEEP].tolist())
    top5_frozen = set(FROZEN[:TOP_KEEP].tolist())
    top72_own = set(orders[L][:G2_END].tolist())
    top72_frozen = set(FROZEN[:G2_END].tolist())
    kept5 = len(top5_own & top5_frozen) / TOP_KEEP
    kept72 = len(top72_own & top72_frozen) / G2_END
    # dangerous direction: channels this layer considers loud that the frozen
    # order does NOT protect (they land in mid or bulk and drag a shared scale)
    missed = len([c for c in top72_own if frozen_groups[c].item() >= 2])
    out(f"{L:5d} {100*same:11.2f}% {100*kept5:10.1f}% {100*kept72:11.1f}% "
        f"{missed:13d}")
    drift[L] = {"same_group_frac": same, "top5_kept": kept5,
                "top72_kept": kept72, "loud_channels_missed": missed}
out()

# ---- quality with frozen vs per-layer order ----
out("=" * 78)
out("QUALITY: FROZEN ORDER vs PER-LAYER ORDER")
out("=" * 78)

results = {"model": MODEL_NAME, "calib_layer": CALIB_LAYER,
           "splits": SPLITS, "ref_perplexity": ref_ppl,
           "drift": drift, "quality": {}, "scheme_changes": []}

for L in SPLITS:
    out(f"--- split L{L} ---")
    out(f"{'scheme':18s} {'bytes':>9s} | {'KL froz':>9s} {'KL own':>9s} "
        f"| {'PPL froz':>9s} {'PPL own':>9s} | {'PPL gain':>9s}")
    out("-" * 78)
    rec = {}
    for scheme in SCHEMES:
        with torch.no_grad():
            lg_f, nb = forward_with_scheme(L, scheme, FROZEN)
            kl_f = F.kl_div((F.softmax(lg_f, -1) + 1e-12).log(), ref_probs,
                            reduction="none").sum(-1).mean().item()
            ppl_f = perplexity(lg_f)

            lg_o, _ = forward_with_scheme(L, scheme, orders[L])
            kl_o = F.kl_div((F.softmax(lg_o, -1) + 1e-12).log(), ref_probs,
                            reduction="none").sum(-1).mean().item()
            ppl_o = perplexity(lg_o)

        rise_f = 100 * (ppl_f - ref_ppl) / ref_ppl
        rise_o = 100 * (ppl_o - ref_ppl) / ref_ppl
        label = f"grouped{scheme[1]}"
        out(f"{label:18s} {nb:9.0f} | {kl_f:9.4f} {kl_o:9.4f} "
            f"| {rise_f:8.2f}% {rise_o:8.2f}% | {rise_f-rise_o:8.2f}pp")
        rec[label] = {"bytes": nb, "kl_frozen": kl_f, "kl_own": kl_o,
                      "ppl_rise_frozen": rise_f, "ppl_rise_own": rise_o}
    results["quality"][str(L)] = rec
    out()

# ---- THE DECIDER: does the scheme pick change? ----
out("=" * 78)
out("SCHEME SELECTED AT EACH BUDGET: FROZEN vs PER-LAYER")
out("=" * 78)
out("Only a CHANGE here saves bytes. Quality gains that do not cross a")
out("scheme threshold cost the same to transmit.")
out()

total_changes = 0
byte_savings = []
for B in PPL_BUDGETS:
    out(f"--- perplexity budget <= {B}% ---")
    out(f"{'layer':>5} {'frozen':>18} {'per-layer':>18} {'bytes saved':>13}")
    out("-" * 60)
    for L in SPLITS:
        rec = results["quality"][str(L)]

        def cheapest(key):
            ok = [(k, v) for k, v in rec.items() if v[key] <= B]
            return min(ok, key=lambda kv: kv[1]["bytes"]) if ok else None

        f_pick = cheapest("ppl_rise_frozen")
        o_pick = cheapest("ppl_rise_own")
        f_lbl = f_pick[0] if f_pick else "none"
        o_lbl = o_pick[0] if o_pick else "none"
        if f_pick and o_pick:
            saved = f_pick[1]["bytes"] - o_pick[1]["bytes"]
        else:
            saved = 0
        mark = ""
        if f_lbl != o_lbl:
            mark = "  <-- CHANGED"
            total_changes += 1
            if saved > 0:
                byte_savings.append(100 * saved / f_pick[1]["bytes"])
        out(f"{L:5d} {f_lbl:>18} {o_lbl:>18} {saved:13.0f}{mark}")
        if f_lbl != o_lbl:
            results["scheme_changes"].append(
                {"budget": B, "layer": L, "frozen": f_lbl, "own": o_lbl,
                 "bytes_saved": saved})
    out()

out("=" * 78)
out("OUTCOME")
out("=" * 78)
out(f"Scheme selection changed in {total_changes} of "
    f"{len(PPL_BUDGETS) * len(SPLITS)} (budget, layer) cases.")
if byte_savings:
    out(f"Where it changed and saved bytes: {min(byte_savings):.1f}% to "
        f"{max(byte_savings):.1f}% reduction "
        f"(mean {sum(byte_savings)/len(byte_savings):.1f}%).")
else:
    out("No case where per-layer ordering reduced the transmitted size.")
out()
out("Decision rule from the experiment plan:")
out("  saves 3-5%  -> implement per-layer ordering, it is free at runtime")
out("  saves ~0.2% -> do not implement; report that a single frozen order is")
out("                 sufficient across the feasible split range, which")
out("                 simplifies the system")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.json")
print(f"Saved: {OUT_DIR}/{RUN_TAG}.txt")
