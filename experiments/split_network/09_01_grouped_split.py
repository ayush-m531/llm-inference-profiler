import torch
import time
import json
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LENS = [5, 50, 100, 250]
KL_THRESHOLD = 1.0
BANDWIDTHS_MBPS = [1, 10, 100]
SLOWDOWNS = [1, 20]
NUM_TIMING_RUNS = 5

# group boundaries (from experiment 05_04)
TOP_KEEP = 10        # channels 1-10 kept full precision
G2_END = 40          # channels 11-40  = "shoulder"
G3_END = 150         # channels 41-150 = "mid"
# channels 151+ = "bulk"

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
               "by allowing models to attend to all positions in a sequence simultaneously. "
               "This led directly to the large language models that power modern systems, "
               "which demonstrate surprising capabilities across many different tasks. ") * 3


def quant_group(x, n_bits):
    """Quantize one group with its own scale."""
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
    """Baseline: one scale for the whole tensor."""
    x = hidden.float()
    if n_bits >= 16:
        return x.clone(), x.shape[1] * x.shape[2] * 16 / 8
    out = quant_group(x.flatten(), n_bits).reshape(x.shape)
    nbytes = (x.shape[1] * x.shape[2] * n_bits) / 8 + 4
    return out, nbytes


def apply_grouped(hidden, bits_combo):
    """Group-wise: top-10 full precision, then 3 groups each with own scale.
    bits_combo = (shoulder_bits, mid_bits, bulk_bits)"""
    x = hidden.float()
    seq_len, dim = x.shape[1], x.shape[2]

    # rank channels by magnitude
    per_channel_max = x.abs().max(dim=1).values.flatten()
    order = torch.argsort(per_channel_max, descending=True)

    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:G2_END]
    g3_idx = order[G2_END:G3_END]
    g4_idx = order[G3_END:]

    out = x.clone()
    out[..., top_idx] = x[..., top_idx]                      # full precision
    b2, b3, b4 = bits_combo
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        out[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)

    # data cost: full-precision top channels + each quantized group + one scale each
    nbytes = (TOP_KEEP * seq_len * 16) / 8
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        nbytes += (len(idx) * seq_len * b) / 8 + 4
    return out, nbytes


def transfer_ms(nbytes, bw_mbps):
    return ((nbytes * 8) / (bw_mbps * 1_000_000)) * 1000


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. Layers: {n_layers}\n")


def run_with_scheme(ids, quant_split, scheme):
    """Run full model; at quant_split, apply the given compression scheme.
    scheme = ('uniform', bits) or ('grouped', (b2,b3,b4)) or ('none', None)
    Returns (logits, nbytes_at_split)."""
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


# candidate schemes, ordered cheapest -> most expensive (we pick the FIRST that meets quality)
SCHEMES = [
    ("grouped", (4, 4, 4)),    # cheapest grouped
    ("uniform", 4),
    ("grouped", (8, 4, 4)),    # the 05_04 "best value"
    ("grouped", (8, 8, 4)),
    ("uniform", 8),
    ("grouped", (8, 8, 8)),
    ("uniform", 16),           # fallback: always works
]

all_tokens = tokenizer(LONG_PROMPT, return_tensors="pt")["input_ids"][0]
splits = list(range(1, n_layers))
results = {"seq_lens": SEQ_LENS, "kl_threshold": KL_THRESHOLD, "data": []}

for seq_len in SEQ_LENS:
    ids = all_tokens[:seq_len].unsqueeze(0).to("cuda")
    print(f"\n{'='*75}\nSEQ LEN = {seq_len}\n{'='*75}")

    with torch.no_grad():
        _ = run_with_scheme(ids, None, ("none", None))   # warmup

    # measure compute per split
    edge_ms = [0.0] * n_layers
    cloud_ms = [0.0] * n_layers
    with torch.no_grad():
        for _ in range(NUM_TIMING_RUNS):
            for s in splits:
                h = model.model.embed_tokens(ids)
                pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
                pe = model.model.rotary_emb(h, pos)
                torch.cuda.synchronize(); t0 = time.time()
                for i in range(s):
                    h = model.model.layers[i](h, attention_mask=None, position_ids=pos,
                        position_embeddings=pe, past_key_values=None, use_cache=False)
                torch.cuda.synchronize(); t1 = time.time()
                for i in range(s, n_layers):
                    h = model.model.layers[i](h, attention_mask=None, position_ids=pos,
                        position_embeddings=pe, past_key_values=None, use_cache=False)
                torch.cuda.synchronize(); t2 = time.time()
                edge_ms[s] += (t1 - t0) * 1000
                cloud_ms[s] += (t2 - t1) * 1000
    edge_ms = [e / NUM_TIMING_RUNS for e in edge_ms]
    cloud_ms = [c / NUM_TIMING_RUNS for c in cloud_ms]

    # for each split: find CHEAPEST scheme meeting quality
    with torch.no_grad():
        ref, _ = run_with_scheme(ids, None, ("none", None))
        ref_p = F.softmax(ref, dim=-1)
        best_scheme = {}
        split_bytes = {}
        for s in splits:
            chosen = None
            for scheme in SCHEMES:
                logits, nb = run_with_scheme(ids, s, scheme)
                dmg = F.softmax(logits, dim=-1)
                kl = F.kl_div((dmg + 1e-12).log(), ref_p, reduction="sum").item()
                if kl <= KL_THRESHOLD:
                    chosen = (scheme, nb, kl)
                    break
            if chosen is None:
                logits, nb = run_with_scheme(ids, s, ("uniform", 16))
                chosen = (("uniform", 16), nb, 0.0)
            best_scheme[s] = chosen[0]
            split_bytes[s] = chosen[1]

    print("\n  Cheapest scheme meeting quality, per split:")
    for s in splits:
        kind, param = best_scheme[s]
        label = f"{kind}{param}"
        print(f"    L{s:2d}: {label:20s} -> {split_bytes[s]:9.0f} bytes")

    seq_rec = {"seq_len": seq_len,
               "best_scheme": {str(s): [best_scheme[s][0], str(best_scheme[s][1])]
                               for s in splits},
               "split_bytes": {str(s): split_bytes[s] for s in splits},
               "scenarios": []}

    print("\n  Optimal split per scenario:")
    for slowdown in SLOWDOWNS:
        for bw in BANDWIDTHS_MBPS:
            totals = {}
            for s in splits:
                totals[s] = (edge_ms[s] * slowdown
                             + transfer_ms(split_bytes[s], bw)
                             + cloud_ms[s])
            best_s = min(totals, key=totals.get)
            k, p = best_scheme[best_s]
            seq_rec["scenarios"].append({
                "bandwidth_mbps": bw, "slowdown": slowdown,
                "best_split": best_s, "best_total_ms": totals[best_s],
                "scheme": f"{k}{p}", "bytes": split_bytes[best_s],
            })
            print(f"    {bw:3} Mbps, x{slowdown:<2} -> SPLIT {best_s:2d} | "
                  f"{k}{p} | {split_bytes[best_s]:7.0f} B | total {totals[best_s]:8.1f} ms")
    results["data"].append(seq_rec)

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/09_01_grouped_split.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to {results_path}")

plt.figure(figsize=(12, 6))
for slowdown in SLOWDOWNS:
    for bw in BANDWIDTHS_MBPS:
        xs, ys = [], []
        for rec in results["data"]:
            for sc in rec["scenarios"]:
                if sc["bandwidth_mbps"] == bw and sc["slowdown"] == slowdown:
                    xs.append(rec["seq_len"])
                    ys.append(sc["best_split"])
        plt.plot(xs, ys, marker="o", label=f"{bw}Mbps x{slowdown}")
plt.xlabel("Sequence length (tokens)")
plt.ylabel("Optimal split point (layer)")
plt.title(f"Optimal split with GROUP-WISE quantization (KL <= {KL_THRESHOLD})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/09_01_grouped_split.png", dpi=150)
print("Plot saved.")
