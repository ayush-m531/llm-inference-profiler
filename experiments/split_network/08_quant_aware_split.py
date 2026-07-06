import torch
import time
import json
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
NUM_TIMING_RUNS = 10
BIT_WIDTHS = [16, 8, 4, 3]          # 16 = full precision baseline
BANDWIDTHS_MBPS = [1, 10, 100]
SLOWDOWN_FACTORS = [1, 20, 50]      # 1 = A100-as-edge baseline, 20/50 = weak edge

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/split_network/results"


def quantize_dequantize(x, n_bits):
    if n_bits >= 16:
        return x  # no quantization
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)
    return q * scale


def data_bytes(num_elements, n_bits):
    # bits -> bytes; add small constant for scale metadata (4 bytes)
    return (num_elements * n_bits) / 8 + 4


def transfer_ms(nbytes, bandwidth_mbps):
    bits = nbytes * 8
    return (bits / (bandwidth_mbps * 1_000_000)) * 1000


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. Layers: {n_layers}")

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")


def run_to_logits(quant_split=None, n_bits=16):
    # full forward pass; if quant_split set, quantize that layer's output
    hidden = model.model.embed_tokens(input_ids)
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None, use_cache=False)
        if quant_split is not None and i == quant_split:
            od = hidden.dtype
            hidden = quantize_dequantize(hidden.float(), n_bits).to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0, -1, :].float()


# warmup
print("Warming up...")
with torch.no_grad():
    _ = run_to_logits(None, 16)
print("Warmup done.\n")

# reference logits (no quantization)
with torch.no_grad():
    ref_logits = run_to_logits(None, 16)
    ref_probs = F.softmax(ref_logits, dim=-1)

# --- measure per-split compute times (averaged) ---
print("Measuring compute times...")
edge_ms = [0.0] * n_layers
cloud_ms = [0.0] * n_layers
act_elems = None
with torch.no_grad():
    for _ in range(NUM_TIMING_RUNS):
        for split in range(1, n_layers):
            hidden = model.model.embed_tokens(input_ids)
            pos = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
            pos_emb = model.model.rotary_emb(hidden, pos)
            torch.cuda.synchronize(); t0 = time.time()
            for i in range(split):
                hidden = model.model.layers[i](hidden, attention_mask=None, position_ids=pos,
                    position_embeddings=pos_emb, past_key_values=None, use_cache=False)
            torch.cuda.synchronize(); t1 = time.time()
            if act_elems is None:
                act_elems = hidden.nelement()
            for i in range(split, n_layers):
                hidden = model.model.layers[i](hidden, attention_mask=None, position_ids=pos,
                    position_embeddings=pos_emb, past_key_values=None, use_cache=False)
            torch.cuda.synchronize(); t2 = time.time()
            edge_ms[split] += (t1 - t0) * 1000
            cloud_ms[split] += (t2 - t1) * 1000
edge_ms = [e / NUM_TIMING_RUNS for e in edge_ms]
cloud_ms = [c / NUM_TIMING_RUNS for c in cloud_ms]

# --- measure quality (KL) for each (split, bits) ---
print("Measuring quantization quality per (split, bit-width)...\n")
kl_table = {}  # (split, bits) -> kl
with torch.no_grad():
    for split in range(1, n_layers):
        for b in BIT_WIDTHS:
            if b >= 16:
                kl_table[(split, b)] = 0.0
                continue
            dmg_logits = run_to_logits(quant_split=split, n_bits=b)
            dmg_probs = F.softmax(dmg_logits, dim=-1)
            kl = F.kl_div((dmg_probs + 1e-12).log(), ref_probs, reduction="sum").item()
            kl_table[(split, b)] = kl

# --- combine: for each scenario, find best (split, bits) by total latency ---
splits = list(range(1, n_layers))
results = {"scenarios": []}

for slowdown in SLOWDOWN_FACTORS:
    for bw in BANDWIDTHS_MBPS:
        best = None
        grid = []
        for s in splits:
            for b in BIT_WIDTHS:
                nbytes = data_bytes(act_elems, b)
                total = edge_ms[s] * slowdown + transfer_ms(nbytes, bw) + cloud_ms[s]
                kl = kl_table[(s, b)]
                grid.append({"split": s, "bits": b, "total_ms": total, "kl": kl,
                             "data_bytes": nbytes})
                if best is None or total < best["total_ms"]:
                    best = {"split": s, "bits": b, "total_ms": total, "kl": kl}
        results["scenarios"].append({
            "slowdown": slowdown, "bandwidth_mbps": bw,
            "best": best, "grid": grid
        })
        print(f"slowdown x{slowdown:<3} | {bw:3} Mbps -> best: split {best['split']:2d}, "
              f"{best['bits']:2d}-bit | total {best['total_ms']:7.2f} ms | KL {best['kl']:.4f}")

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/08_quant_aware_split.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

# plot: best split vs bandwidth, one line per slowdown
plt.figure(figsize=(12, 6))
for slowdown in SLOWDOWN_FACTORS:
    xs, ys = [], []
    for sc in results["scenarios"]:
        if sc["slowdown"] == slowdown:
            xs.append(sc["bandwidth_mbps"])
            ys.append(sc["best"]["split"])
    plt.plot(xs, ys, marker="o", label=f"slowdown x{slowdown}")
plt.xscale("log")
plt.xlabel("Bandwidth (Mbps, log scale)")
plt.ylabel("Best split point (layer)")
plt.title("Optimal Split Point vs Bandwidth, across Edge Slowdown Factors")
plt.legend()
plt.tight_layout()
plot_path = f"{RESULTS_DIR}/08_quant_aware_split.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
