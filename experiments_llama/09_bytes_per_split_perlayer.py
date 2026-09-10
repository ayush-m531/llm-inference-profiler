"""
Bytes per split, recomputed with PER-LAYER channel orders.

Exp 03 measured a 24.1% saving for splitting in the middle rather than at
layer 0/1, using ONE frozen channel order taken from layer 2. Exp 07/08 then
showed per-layer orders are better and should be adopted. Those two results
are in conflict: under per-layer orders, layer 0 stops being expensive, so
there may be nothing left to save by splitting later.

This measures both orderings side by side on the same run, so the comparison
is direct rather than inferred from another experiment's KL table.

Orders are built from a SEPARATE calibration text, not from the evaluation
text, matching the deployable design validated in Exp 08.

Writes JSON and a TXT transcript.
"""
import torch
import json
import os
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "09_bytes_per_split_perlayer"

SEQ_LEN = 100
KL_THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 1.0]

TOP_KEEP = 5
G2_END = 72
G3_END = 1312
CALIB_LAYER = 2          # source of the single frozen order

# feasible range from the CORRECTED memory ceiling (see note in results):
# 5.90 GB free -> 7 layers fit -> deepest split L6
# 7.81 GB free -> 11 layers fit -> deepest split L10
FEASIBLE_LO, FEASIBLE_HI = 5, 10

# Orders come from THIS text, evaluation happens on the one below.
CALIB_TEXT = ("A recipe for a simple loaf begins with flour, water, salt and yeast. "
              "The baker mixes them until a rough dough forms, then rests it so the "
              "flour can absorb the water fully. Kneading develops the gluten network "
              "that traps gas during fermentation. After the first rise the dough is "
              "shaped and left to prove a second time, more briefly. A hot oven with "
              "steam in the early minutes lets the loaf expand before the crust sets. "
              "Bakers judge doneness by colour and by tapping the base, listening for "
              "a hollow sound. Cooling on a rack prevents the trapped steam from "
              "softening the crust. Variations in hydration, flour type and "
              "fermentation time produce very different results from the same four "
              "ingredients, which is why bread rewards practice more than precision. ") * 4

EVAL_TEXT = ("The history of artificial intelligence began in the 1950s when researchers "
             "started exploring the possibility of creating machines that could think and "
             "reason like humans. Early pioneers developed symbolic systems and search "
             "algorithms, believing that intelligence could be captured through logical "
             "rules. Over the following decades the field experienced cycles of optimism "
             "and disappointment, often called AI winters, as early promises failed to "
             "materialize. The introduction of machine learning shifted the paradigm from "
             "hand crafted rules toward systems that learn patterns directly from data. "
             "Neural networks, inspired loosely by biological brains, gradually became the "
             "dominant approach, especially after advances in computing hardware made it "
             "practical to train very large models on enormous datasets. The transformer "
             "architecture, introduced in 2017, revolutionized natural language processing "
             "by allowing models to attend to all positions in a sequence simultaneously. ") * 3


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


def apply_uniform(hidden, n_bits):
    x = hidden.float()
    if n_bits >= 16:
        return x.clone(), x.shape[1] * x.shape[2] * 16 / 8
    o = quant_group(x.flatten(), n_bits).reshape(x.shape)
    return o, (x.shape[1] * x.shape[2] * n_bits) / 8 + 4


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


# ordered CHEAPEST FIRST by actual byte cost, not by hand
SCHEMES = [
    ("uniform", 4),
    ("grouped", (4, 4, 4)),
    ("grouped", (8, 4, 4)),
    ("grouped", (8, 8, 4)),
    ("uniform", 8),
    ("grouped", (8, 8, 8)),
    ("uniform", 16),
]

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
H = model.config.hidden_size
splits = list(range(0, n_layers))


def build_orders(text):
    ids = tok(text, return_tensors="pt")["input_ids"][0][:SEQ_LEN].unsqueeze(0).to("cuda")
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    o = {}
    for L in range(n_layers):
        act = hs[L + 1][0].float()
        o[L] = torch.argsort(act.abs().max(dim=0).values, descending=True).to("cuda")
    del hs
    return o


print("Building per-layer orders from calibration text...")
orders = build_orders(CALIB_TEXT)
FROZEN = orders[CALIB_LAYER]

ids = tok(EVAL_TEXT, return_tensors="pt")["input_ids"][0][:SEQ_LEN].unsqueeze(0).to("cuda")


def run(split, scheme, order):
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    nbytes = None
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None,
                       use_cache=False)
        if split is not None and i == split:
            kind, param = scheme
            od = hidden.dtype
            if kind == "uniform":
                q, nbytes = apply_uniform(hidden, param)
            else:
                q, nbytes = apply_grouped(hidden, param, order)
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0].float(), nbytes


with torch.no_grad():
    ref_logits, _ = run(None, None, FROZEN)
    ref_probs = F.softmax(ref_logits, dim=-1)


def kl_all(lg):
    p = F.softmax(lg, dim=-1)
    return F.kl_div((p + 1e-12).log(), ref_probs,
                    reduction="none").sum(-1).mean().item()


out(f"Model: {MODEL_NAME}")
out(f"seq_len {SEQ_LEN} | orders from calibration text (not the eval text)")
out(f"Feasible split range (corrected memory ceiling): L{FEASIBLE_LO}-L{FEASIBLE_HI}")
out(f"Boundaries: top={TOP_KEEP}, shoulder<={G2_END}, mid<={G3_END}")
out()

# ---- measure every (split, scheme) under BOTH orderings ----
print("Measuring 32 splits x 7 schemes x 2 orderings...")
kl_f, kl_p, by = {}, {}, {}
with torch.no_grad():
    for s in splits:
        for si, sch in enumerate(SCHEMES):
            lg, nb = run(s, sch, FROZEN)
            kl_f[(s, si)] = kl_all(lg)
            by[(s, si)] = nb
            lg2, _ = run(s, sch, orders[s])
            kl_p[(s, si)] = kl_all(lg2)
        print(f"  split {s:2d} done")

results = {"model": MODEL_NAME, "seq_len": SEQ_LEN,
           "feasible": [FEASIBLE_LO, FEASIBLE_HI],
           "schemes": [f"{k}{v}" for k, v in SCHEMES],
           "thresholds": {}}


def cheapest(s, table, T):
    for si in range(len(SCHEMES)):
        if table[(s, si)] <= T:
            return si
    return len(SCHEMES) - 1


out("=" * 78)
out("BYTES PER SPLIT: FROZEN ORDER vs PER-LAYER ORDERS")
out("=" * 78)

for T in KL_THRESHOLDS:
    bf = {s: by[(s, cheapest(s, kl_f, T))] for s in splits}
    bp = {s: by[(s, cheapest(s, kl_p, T))] for s in splits}
    sf = {s: SCHEMES[cheapest(s, kl_f, T)] for s in splits}
    sp = {s: SCHEMES[cheapest(s, kl_p, T)] for s in splits}

    out(f"\n--- KL <= {T} ---")
    out(f"{'layer':>5} {'frozen bytes':>14} {'per-layer bytes':>17}  "
        f"{'frozen scheme':>16} {'per-layer scheme':>18}")
    out("-" * 78)
    for s in list(range(0, 13)) + [16, 20, 24, 28, 31]:
        out(f"{s:5d} {bf[s]:14.0f} {bp[s]:17.0f}  "
            f"{str(sf[s][1]):>16} {str(sp[s][1]):>18}")

    def saving(b):
        early = min(b[0], b[1])
        mid = min(b[s] for s in range(FEASIBLE_LO, FEASIBLE_HI + 1))
        return early, mid, 100 * (early - mid) / early

    ef, mf, gf = saving(bf)
    ep, mp, gp = saving(bp)
    out("")
    out(f"  FROZEN    : early(L0/L1) {ef:9.0f} | best mid(L{FEASIBLE_LO}-{FEASIBLE_HI}) "
        f"{mf:9.0f} | saving {gf:6.2f}%")
    out(f"  PER-LAYER : early(L0/L1) {ep:9.0f} | best mid(L{FEASIBLE_LO}-{FEASIBLE_HI}) "
        f"{mp:9.0f} | saving {gp:6.2f}%")
    if gp < 1.0:
        out(f"  => under the deployable design the split saving is ~0. Bytes do NOT")
        out(f"     select the split point.")
    else:
        out(f"  => the split saving SURVIVES per-layer ordering.")

    results["thresholds"][str(T)] = {
        "frozen": {"bytes": {str(s): bf[s] for s in splits},
                   "scheme": {str(s): f"{sf[s][0]}{sf[s][1]}" for s in splits},
                   "early": ef, "mid": mf, "saving_pct": gf},
        "per_layer": {"bytes": {str(s): bp[s] for s in splits},
                      "scheme": {str(s): f"{sp[s][0]}{sp[s][1]}" for s in splits},
                      "early": ep, "mid": mp, "saving_pct": gp}}

# full KL table saved so this can be re-audited without a re-run
results["kl_table"] = {
    f"{s}_{si}": {"kl_frozen": kl_f[(s, si)], "kl_perlayer": kl_p[(s, si)],
                  "bytes": by[(s, si)]}
    for s in splits for si in range(len(SCHEMES))}

out("\n" + "=" * 78)
out("SUMMARY")
out("=" * 78)
for T in KL_THRESHOLDS:
    r = results["thresholds"][str(T)]
    out(f"  KL<={T:<5}: frozen {r['frozen']['saving_pct']:6.2f}%   "
        f"per-layer {r['per_layer']['saving_pct']:6.2f}%")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")

plt.figure(figsize=(13, 6))
for T in [0.05, 0.1]:
    r = results["thresholds"][str(T)]
    plt.plot(splits, [r["frozen"]["bytes"][str(s)] for s in splits],
             marker="o", linestyle="--", label=f"frozen, KL<={T}")
    plt.plot(splits, [r["per_layer"]["bytes"][str(s)] for s in splits],
             marker=".", label=f"per-layer, KL<={T}")
plt.axvspan(FEASIBLE_LO, FEASIBLE_HI, alpha=0.12, color="green")
plt.xlabel("Split point (layer)")
plt.ylabel("Bytes needed to meet quality budget")
plt.title("Bytes per split: frozen vs per-layer channel orders")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, f"{RUN_TAG}.png"), dpi=150)
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt,png}}")
