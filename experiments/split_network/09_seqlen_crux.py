import torch
import time
import json
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
SEQ_LENS = [5, 50, 100, 250]
BIT_OPTIONS = [3, 4, 8, 16]        # ascending: we pick the SMALLEST that meets quality
KL_THRESHOLD = 1.0                 # quality budget
BANDWIDTHS_MBPS = [1, 10, 100]
SLOWDOWNS = [1, 20]
NUM_TIMING_RUNS = 5

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/split_network/results"

# a long prompt we can truncate to each seq_len
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


def quantize_dequantize(x, n_bits):
    if n_bits >= 16:
        return x
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def transfer_ms(nbytes, bw_mbps):
    return ((nbytes * 8) / (bw_mbps * 1_000_000)) * 1000


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
hidden_dim = model.config.hidden_size
print(f"Model loaded. Layers: {n_layers}, hidden dim: {hidden_dim}\n")


def run_to_logits(ids, quant_split=None, n_bits=16):
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None, use_cache=False)
        if quant_split is not None and i == quant_split:
            od = hidden.dtype
            hidden = quantize_dequantize(hidden.float(), n_bits).to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0, -1, :].float()


all_tokens = tokenizer(LONG_PROMPT, return_tensors="pt")["input_ids"][0]
splits = list(range(1, n_layers))
results = {"seq_lens": SEQ_LENS, "kl_threshold": KL_THRESHOLD, "data": []}

for seq_len in SEQ_LENS:
    ids = all_tokens[:seq_len].unsqueeze(0).to("cuda")
    print(f"\n{'='*70}\nSEQ LEN = {seq_len}\n{'='*70}")

    # warmup
    with torch.no_grad():
        _ = run_to_logits(ids)

    # --- measure compute per split (real, at this seq len) ---
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

    # --- for each split: find MINIMUM bits that keeps KL <= threshold ---
    with torch.no_grad():
        ref = F.softmax(run_to_logits(ids), dim=-1)
        min_bits = {}
        kl_at_min = {}
        for s in splits:
            chosen = None
            for b in BIT_OPTIONS:          # ascending: 3, 4, 8, 16
                dmg = F.softmax(run_to_logits(ids, s, b), dim=-1)
                kl = F.kl_div((dmg + 1e-12).log(), ref, reduction="sum").item()
                if kl <= KL_THRESHOLD:
                    chosen = (b, kl)
                    break
            if chosen is None:
                chosen = (16, 0.0)          # fallback: full precision always works
            min_bits[s] = chosen[0]
            kl_at_min[s] = chosen[1]

    n_elems = seq_len * hidden_dim
    print(f"\n  Minimum bits needed per split (KL <= {KL_THRESHOLD}):")
    print("  " + " ".join(f"L{s}:{min_bits[s]}b" for s in splits))

    # --- total latency per scenario ---
    seq_rec = {"seq_len": seq_len, "min_bits": min_bits, "scenarios": []}
    for slowdown in SLOWDOWNS:
        for bw in BANDWIDTHS_MBPS:
            totals = {}
            for s in splits:
                nbytes = (n_elems * min_bits[s]) / 8 + 4
                totals[s] = (edge_ms[s] * slowdown
                             + transfer_ms(nbytes, bw)
                             + cloud_ms[s])
            best_s = min(totals, key=totals.get)
            seq_rec["scenarios"].append({
                "bandwidth_mbps": bw, "slowdown": slowdown,
                "best_split": best_s, "best_total_ms": totals[best_s],
                "best_bits": min_bits[best_s],
                "totals": totals,
            })
            print(f"    {bw:3} Mbps, x{slowdown:<2} -> BEST SPLIT = {best_s:2d} "
                  f"({min_bits[best_s]}-bit) | total {totals[best_s]:8.1f} ms")
    results["data"].append(seq_rec)

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/09_seqlen_crux.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to {results_path}")

# --- plot: best split vs seq_len, one line per (bw, slowdown) ---
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
plt.title(f"Does the optimal split move off layer 1 as sequence length grows?\n"
          f"(quality held fixed: KL <= {KL_THRESHOLD})")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/09_seqlen_crux.png", dpi=150)
print("Plot saved.")
