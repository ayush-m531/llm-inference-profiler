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
BIT_WIDTHS = [16, 8, 4, 3]
# scenarios: (bandwidth_mbps, slowdown_factor)
SCENARIOS = [(1, 1), (1, 50), (100, 1), (100, 50)]
KL_THRESHOLDS = [0.5, 1.0, 2.0]

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/split_network/results"


def quantize_dequantize(x, n_bits):
    if n_bits >= 16:
        return x
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)
    return q * scale


def data_bytes(num_elements, n_bits):
    return (num_elements * n_bits) / 8 + 4


def transfer_ms(nbytes, bandwidth_mbps):
    return ((nbytes * 8) / (bandwidth_mbps * 1_000_000)) * 1000


def pareto_frontier(options):
    # keep options not dominated by any other (lower is better on both latency and kl)
    frontier = []
    for a in options:
        dominated = False
        for b in options:
            if b is a:
                continue
            if b["total_ms"] <= a["total_ms"] and b["kl"] <= a["kl"] and \
               (b["total_ms"] < a["total_ms"] or b["kl"] < a["kl"]):
                dominated = True
                break
        if not dominated:
            frontier.append(a)
    frontier.sort(key=lambda o: o["total_ms"])
    return frontier


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


print("Warming up...")
with torch.no_grad():
    _ = run_to_logits(None, 16)
print("Warmup done.\n")

with torch.no_grad():
    ref_logits = run_to_logits(None, 16)
    ref_probs = F.softmax(ref_logits, dim=-1)

# compute timings
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

# quality per (split, bits)
print("Measuring quality per (split, bit-width)...\n")
kl_table = {}
with torch.no_grad():
    for split in range(1, n_layers):
        for b in BIT_WIDTHS:
            if b >= 16:
                kl_table[(split, b)] = 0.0
                continue
            dmg_logits = run_to_logits(quant_split=split, n_bits=b)
            dmg_probs = F.softmax(dmg_logits, dim=-1)
            kl_table[(split, b)] = F.kl_div((dmg_probs + 1e-12).log(), ref_probs,
                                            reduction="sum").item()

splits = list(range(1, n_layers))
results = {"scenarios": []}

for bw, slowdown in SCENARIOS:
    options = []
    for s in splits:
        for b in BIT_WIDTHS:
            nbytes = data_bytes(act_elems, b)
            total = edge_ms[s] * slowdown + transfer_ms(nbytes, bw) + cloud_ms[s]
            options.append({"split": s, "bits": b, "total_ms": total,
                            "kl": kl_table[(s, b)]})

    frontier = pareto_frontier(options)

    picks = {}
    for T in KL_THRESHOLDS:
        valid = [o for o in options if o["kl"] <= T]
        picks[T] = min(valid, key=lambda o: o["total_ms"]) if valid else None

    results["scenarios"].append({
        "bandwidth_mbps": bw, "slowdown": slowdown,
        "frontier": frontier, "threshold_picks": {str(k): v for k, v in picks.items()},
    })

    print(f"=== {bw} Mbps, slowdown x{slowdown} ===")
    print("  Pareto frontier (fastest -> highest quality):")
    for o in frontier:
        print(f"    split {o['split']:2d}, {o['bits']:2d}-bit | "
              f"{o['total_ms']:8.2f} ms | KL {o['kl']:.4f}")
    print("  Threshold picks:")
    for T in KL_THRESHOLDS:
        p = picks[T]
        if p:
            print(f"    KL<={T}: split {p['split']:2d}, {p['bits']:2d}-bit | "
                  f"{p['total_ms']:8.2f} ms | KL {p['kl']:.4f}")
        else:
            print(f"    KL<={T}: no option meets this quality")
    print()

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/08_01_controller.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")

fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(5 * len(SCENARIOS), 5), squeeze=False)
for idx, (bw, slowdown) in enumerate(SCENARIOS):
    ax = axes[0][idx]
    sc = results["scenarios"][idx]
    fr = sc["frontier"]
    ax.plot([o["total_ms"] for o in fr], [o["kl"] for o in fr],
            marker="o", color="steelblue")
    for o in fr:
        ax.annotate(f"s{o['split']},{o['bits']}b",
                    (o["total_ms"], o["kl"]), fontsize=7)
    for T in KL_THRESHOLDS:
        ax.axhline(T, linestyle="--", alpha=0.4)
    ax.set_title(f"{bw} Mbps, x{slowdown}")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("KL divergence")
plt.tight_layout()
plot_path = f"{RESULTS_DIR}/08_01_controller.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
