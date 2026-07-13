import torch
import json
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LEN = 100                              # fixed; the KL budget is what we sweep
KL_THRESHOLDS = [0.05, 0.1, 0.25, 0.5, 1.0]

TOP_KEEP = 10
G2_END = 40
G3_END = 150

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/split_network/results"

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


def apply_grouped(hidden, bits_combo):
    x = hidden.float()
    seq_len = x.shape[1]
    per_channel_max = x.abs().max(dim=1).values.flatten()
    order = torch.argsort(per_channel_max, descending=True)

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


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. Layers: {n_layers}\n")


def run_with_scheme(ids, quant_split, scheme):
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
                q, nbytes = apply_grouped(hidden, param)
            else:
                q, nbytes = hidden.float(), hidden.shape[1] * hidden.shape[2] * 16 / 8
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0, -1, :].float(), nbytes


SCHEMES = [
    ("grouped", (4, 4, 4)),
    ("uniform", 4),
    ("grouped", (8, 4, 4)),
    ("grouped", (8, 8, 4)),
    ("uniform", 8),
    ("grouped", (8, 8, 8)),
    ("uniform", 16),
]

all_tokens = tokenizer(LONG_PROMPT, return_tensors="pt")["input_ids"][0]
ids = all_tokens[:SEQ_LEN].unsqueeze(0).to("cuda")
splits = list(range(1, n_layers))

# ---- measure KL for EVERY (split, scheme) once; then apply thresholds ----
print(f"Measuring KL for all (split, scheme) pairs at seq_len={SEQ_LEN}...")
with torch.no_grad():
    _ = run_with_scheme(ids, None, ("none", None))   # warmup
    ref, _ = run_with_scheme(ids, None, ("none", None))
    ref_p = F.softmax(ref, dim=-1)

    kl_table = {}     # (split, scheme_idx) -> kl
    bytes_table = {}  # (split, scheme_idx) -> bytes
    for s in splits:
        for si, scheme in enumerate(SCHEMES):
            logits, nb = run_with_scheme(ids, s, scheme)
            dmg = F.softmax(logits, dim=-1)
            kl = F.kl_div((dmg + 1e-12).log(), ref_p, reduction="sum").item()
            kl_table[(s, si)] = kl
            bytes_table[(s, si)] = nb
        print(f"  split {s:2d} done")

# ---- report raw KL per scheme for a few representative layers ----
print("\n\n=== RAW KL per scheme (representative layers) ===")
print(f"{'scheme':22s}" + "".join(f"{'L'+str(s):>10s}" for s in [1, 3, 10, 20, 22]))
print("-" * 75)
for si, scheme in enumerate(SCHEMES):
    label = f"{scheme[0]}{scheme[1]}"
    row = f"{label:22s}"
    for s in [1, 3, 10, 20, 22]:
        row += f"{kl_table[(s, si)]:10.4f}"
    print(row)

# ---- for each KL threshold: cheapest scheme per split ----
results = {"seq_len": SEQ_LEN, "thresholds": {}}
print("\n\n=== CHEAPEST SCHEME PER SPLIT, AT EACH KL BUDGET ===")
for T in KL_THRESHOLDS:
    print(f"\n--- KL <= {T} ---")
    rec = {}
    for s in splits:
        chosen_i = None
        for si in range(len(SCHEMES)):
            if kl_table[(s, si)] <= T:
                chosen_i = si
                break
        if chosen_i is None:
            chosen_i = len(SCHEMES) - 1   # uniform16 fallback
        scheme = SCHEMES[chosen_i]
        rec[s] = {"scheme": f"{scheme[0]}{scheme[1]}",
                  "bytes": bytes_table[(s, chosen_i)],
                  "kl": kl_table[(s, chosen_i)]}
    # compact print
    line = "  " + " ".join(f"L{s}:{rec[s]['scheme'].replace('grouped','g').replace('uniform','u')}"
                           for s in splits)
    print(line)
    avg_bytes = sum(rec[s]["bytes"] for s in splits) / len(splits)
    storm = [s for s in splits if 2 <= s <= 20]
    storm_cheap = sum(1 for s in storm if rec[s]["scheme"] == "grouped(4, 4, 4)")
    print(f"  -> avg bytes across splits: {avg_bytes:9.0f}")
    print(f"  -> storm layers (2-20) still cheap at grouped(4,4,4): "
          f"{storm_cheap}/{len(storm)}")
    results["thresholds"][str(T)] = {str(s): rec[s] for s in splits}

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/09_02_kl_sweep.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to {RESULTS_DIR}/09_02_kl_sweep.json")

# plot: bytes needed per split, one line per KL threshold
plt.figure(figsize=(13, 6))
for T in KL_THRESHOLDS:
    ys = [results["thresholds"][str(T)][str(s)]["bytes"] for s in splits]
    plt.plot(splits, ys, marker="o", label=f"KL <= {T}")
plt.xlabel("Split point (layer)")
plt.ylabel("Bytes needed to meet quality budget")
plt.title(f"Transmission cost per split point at different quality budgets (seq={SEQ_LEN})")
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(splits)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/09_02_kl_sweep.png", dpi=150)
print("Plot saved.")
