import torch
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- change ONLY this line to switch models ----
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SEQ_LEN = 100
KL_THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 1.0]

# group boundaries for Llama's 4096 channels (from the cliff analysis)
TOP_KEEP = 5        # channels 1-5      full precision
G2_END = 72         # channels 6-72     shoulder
G3_END = 1312       # channels 73-1312  mid
# channels 1313-4096 = bulk

OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"

LONG_PROMPT = ("The history of artificial intelligence began in the 1950s when researchers "
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


def quant_group(x, n_bits):
    """Your exact quantization, unchanged."""
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
    out = quant_group(x.flatten(), n_bits).reshape(x.shape)
    nbytes = (x.shape[1] * x.shape[2] * n_bits) / 8 + 4
    return out, nbytes


def apply_grouped(hidden, bits_combo, fixed_order):
    """Your grouping math, with ONE change: the channel order is passed in
    (frozen once, shipped with the model) instead of recomputed from the
    live tensor. So the receiver can decode it, and it costs 0 runtime bytes."""
    x = hidden.float()
    seq_len = x.shape[1]
    order = fixed_order   # <-- frozen, not argsort'd here

    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:G2_END]
    g3_idx = order[G2_END:G3_END]
    g4_idx = order[G3_END:]

    out = x.clone()
    out[..., top_idx] = x[..., top_idx]
    b2, b3, b4 = bits_combo
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        out[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)

    nbytes = (TOP_KEEP * seq_len * 16) / 8
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        nbytes += (len(idx) * seq_len * b) / 8 + 4
    return out, nbytes


print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print(f"Layers: {n_layers}  hidden: {hidden_size}\n")

all_tokens = tokenizer(LONG_PROMPT, return_tensors="pt")["input_ids"][0]
ids = all_tokens[:SEQ_LEN].unsqueeze(0).to("cuda")
print("Tokens:", ids.shape[1])


# ---- freeze the channel order ONCE, from a calibration pass ----
# We rank channels by magnitude at an early layer and reuse that SAME order
# at every split point. This is what a real deployment does: the order ships
# with the model. Exp 03_01 showed the order is stable across prompts/layers.
print("Building fixed channel order from calibration pass...")
with torch.no_grad():
    calib = model(ids, output_hidden_states=True)
# hidden_states[1] = output of layer 0; use a mid-early layer for a stable ranking
CALIB_LAYER = min(2, n_layers - 1)
calib_act = calib.hidden_states[CALIB_LAYER + 1][0].float()   # [seq, hidden]
per_channel_max = calib_act.abs().max(dim=0).values
FIXED_ORDER = torch.argsort(per_channel_max, descending=True).to("cuda")
del calib, calib_act
print(f"Channel order frozen from layer {CALIB_LAYER}.\n")


def run_with_scheme(quant_split, scheme):
    """Returns (logits_all_positions, nbytes). logits is [seq, vocab]."""
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    nbytes = None
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None, use_cache=False)
        if quant_split is not None and i == quant_split:
            kind, param = scheme
            od = hidden.dtype
            if kind == "uniform":
                q, nbytes = apply_uniform(hidden, param)
            elif kind == "grouped":
                q, nbytes = apply_grouped(hidden, param, FIXED_ORDER)
            else:
                q, nbytes = hidden.float(), hidden.shape[1] * hidden.shape[2] * 16 / 8
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)[0].float()   # [seq, vocab] - ALL positions
    return logits, nbytes


def kl_all_positions(dmg_logits, ref_probs_all):
    """Mean KL over ALL token positions (stricter than last-token only)."""
    dmg = F.softmax(dmg_logits, dim=-1)
    kl = F.kl_div((dmg + 1e-12).log(), ref_probs_all, reduction="none").sum(dim=-1)
    return kl.mean().item()


SCHEMES = [
    ("grouped", (4, 4, 4)),
    ("uniform", 4),
    ("grouped", (8, 4, 4)),
    ("grouped", (8, 8, 4)),
    ("uniform", 8),
    ("grouped", (8, 8, 8)),
    ("uniform", 16),
]

# split 0 INCLUDED this time (range starts at 0, not 1)
splits = list(range(0, n_layers))

print("Measuring KL + bytes for every (split, scheme)...")
with torch.no_grad():
    ref_logits, _ = run_with_scheme(None, ("none", None))
    ref_probs_all = F.softmax(ref_logits, dim=-1)

    kl_table = {}
    bytes_table = {}
    for s in splits:
        for si, scheme in enumerate(SCHEMES):
            logits, nb = run_with_scheme(s, scheme)
            kl_table[(s, si)] = kl_all_positions(logits, ref_probs_all)
            bytes_table[(s, si)] = nb
        print(f"  split {s:2d} done")

# raw KL for a few representative layers
print("\n=== RAW KL per scheme (representative layers) ===")
probe = [0, 1, 5, 15, n_layers // 2, n_layers - 2]
probe = sorted(set(p for p in probe if p in splits))
header = f"{'scheme':22s}" + "".join(f"{'L'+str(s):>10s}" for s in probe)
print(header)
print("-" * len(header))
for si, scheme in enumerate(SCHEMES):
    label = f"{scheme[0]}{scheme[1]}"
    row = f"{label:22s}" + "".join(f"{kl_table[(s, si)]:10.4f}" for s in probe)
    print(row)

# cheapest scheme meeting each budget, per split
results = {"model": MODEL_NAME, "seq_len": int(ids.shape[1]),
           "boundaries": [TOP_KEEP, G2_END, G3_END],
           "thresholds": {}}

print("\n=== CHEAPEST SCHEME PER SPLIT, AT EACH KL BUDGET ===")
for T in KL_THRESHOLDS:
    rec = {}
    for s in splits:
        chosen_i = None
        for si in range(len(SCHEMES)):
            if kl_table[(s, si)] <= T:
                chosen_i = si
                break
        if chosen_i is None:
            chosen_i = len(SCHEMES) - 1
        scheme = SCHEMES[chosen_i]
        rec[s] = {"scheme": f"{scheme[0]}{scheme[1]}",
                  "bytes": bytes_table[(s, chosen_i)],
                  "kl": kl_table[(s, chosen_i)]}
    results["thresholds"][str(T)] = {str(s): rec[s] for s in splits}

    # THE KEY QUESTION printed for every budget
    bvals = {s: rec[s]["bytes"] for s in splits}
    min_bytes = min(bvals.values())
    argmin = [s for s in splits if bvals[s] == min_bytes]
    cheapest_early = min(bvals[0], bvals[1])
    best_later = min(bvals[s] for s in splits if s >= 2)
    best_later_split = min((s for s in splits if s >= 2), key=lambda s: bvals[s])
    print(f"\n--- KL <= {T} ---")
    print(f"  layer 0 bytes: {bvals[0]:9.0f}   layer 1 bytes: {bvals[1]:9.0f}")
    print(f"  cheapest overall: {min_bytes:9.0f} at split(s) {argmin}")
    print(f"  best LATER split (>=2): {best_later:9.0f} at L{best_later_split}")
    if best_later < cheapest_early:
        save = 100 * (cheapest_early - best_later) / cheapest_early
        print(f"  ==> a LATER split is cheaper than layer 0/1 by {save:.1f}%. "
              f"Split axis is LIVE.")
    else:
        print(f"  ==> layer 0/1 is still cheapest. Split axis is FLAT at this budget.")

os.makedirs(OUT_DIR, exist_ok=True)
tag = MODEL_NAME.split("/")[-1]
json_path = os.path.join(OUT_DIR, f"04_bytes_per_split_{tag}.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved:", json_path)

# plot: bytes needed per split, one line per budget
plt.figure(figsize=(13, 6))
for T in KL_THRESHOLDS:
    ys = [results["thresholds"][str(T)][str(s)]["bytes"] for s in splits]
    plt.plot(splits, ys, marker="o", label=f"KL <= {T}")
plt.xlabel("Split point (layer)")
plt.ylabel("Bytes needed to meet quality budget")
plt.title(f"Bytes per split at each quality budget - {tag} (seq={ids.shape[1]})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, f"04_bytes_per_split_{tag}.png")
plt.savefig(png_path, dpi=150)
print("Saved:", png_path)
