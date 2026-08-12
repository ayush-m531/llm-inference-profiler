import torch
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- change ONLY this line to switch models ----
#MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

SEQ_LEN = 100
TOP_K = 15
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"

PROMPT = ("The history of artificial intelligence began in the 1950s when researchers "
          "started exploring the possibility of creating machines that could think and "
          "reason like humans. Early pioneers developed symbolic systems and search "
          "algorithms, believing that intelligence could be captured through logical "
          "rules. Over the following decades the field experienced cycles of optimism "
          "and disappointment, often called AI winters, as early promises failed to "
          "materialize. The introduction of machine learning shifted the paradigm from "
          "hand crafted rules toward systems that learn patterns directly from data. "
          "Neural networks, inspired loosely by biological brains, gradually became the "
          "dominant approach, especially after advances in computing hardware made it "
          "practical to train very large models on enormous datasets. ") * 3


def layer_stats(x):
    """x is [seq_len, hidden] on CPU float32. Returns statistics for one layer."""
    v = x.reshape(-1).double()
    mean = v.mean()
    centered = v - mean
    var = (centered ** 2).mean()
    kurt = ((centered ** 4).mean() / (var ** 2)).item()

    absx = x.abs().double()
    max_abs = absx.max().item()
    mean_abs = absx.mean().item()

    per_ch = absx.max(dim=0).values
    total = per_ch.sum().item()
    n_ch = per_ch.numel()
    k1 = max(1, int(round(n_ch * 0.01)))
    sorted_vals, order = torch.sort(per_ch, descending=True)
    top1pct = sorted_vals[:k1].sum().item() / total if total > 0 else 0.0

    return {
        "kurtosis": kurt,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_over_mean": max_abs / mean_abs if mean_abs > 0 else 0.0,
        "top1pct_share": top1pct,
        "top_channels": order[:TOP_K].tolist(),
    }


print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print("Layers:", n_layers, " hidden size:", hidden_size)

all_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0]
ids = all_ids[:SEQ_LEN].unsqueeze(0).to("cuda")
print("Tokens used:", ids.shape[1])

print("Running one forward pass...")
with torch.no_grad():
    out = model(ids, output_hidden_states=True)

# hidden_states has n_layers + 1 entries.
# entry 0 = embedding output. entry i+1 = output of decoder layer i.
hs = out.hidden_states

rows = []
for i in range(n_layers):
    x = hs[i + 1][0].float().cpu()
    s = layer_stats(x)
    s["layer"] = i
    rows.append(s)
    del x

print()
print("layer      kurtosis       max_abs    max/mean    top1%_share")
print("-" * 62)
for r in rows:
    print(f"{r['layer']:5d} {r['kurtosis']:14.1f} {r['max_abs']:13.1f} "
          f"{r['max_over_mean']:11.1f} {100 * r['top1pct_share']:11.1f}%")

kurts = [r["kurtosis"] for r in rows]
max_k = max(kurts)
max_layer = kurts.index(max_k)

print()
print("=" * 62)
print("THE QUESTION: is layer 1 calm, or is it already stormy?")
print("=" * 62)
print(f"  layer 0 kurtosis  : {kurts[0]:14.1f}")
print(f"  layer 1 kurtosis  : {kurts[1]:14.1f}")
print(f"  biggest kurtosis  : {max_k:14.1f}   (at layer {max_layer})")
print(f"  layer 1 / biggest : {kurts[1] / max_k:.4f}")
print()
if kurts[1] < 0.1 * max_k:
    print("  ANSWER: LAYER 1 IS CALM. Same shape as Qwen.")
    print("  Layer 1 will compress well, so it stays the cheapest place to cut.")
    print("  The quantization-aware split idea gains nothing at this scale.")
else:
    print("  ANSWER: LAYER 1 IS ALREADY STORMY. Different from Qwen.")
    print("  Layer 1 is expensive to send, so a later calm layer can beat it.")
    print("  The quantization-aware split idea is ALIVE at this scale.")

print()
print("Are the outlier channels the same set in every layer?")
mid = n_layers // 2
ref = set(rows[mid]["top_channels"])
probe = sorted(set([1, 2, mid, n_layers - 2]))
for i in probe:
    shared = len(ref & set(rows[i]["top_channels"]))
    print(f"  layer {i:2d}: {shared:2d}/{TOP_K} of its top channels match layer {mid}")

os.makedirs(OUT_DIR, exist_ok=True)
tag = MODEL_NAME.split("/")[-1]
json_path = os.path.join(OUT_DIR, f"03_activation_stats_{tag}.json")
with open(json_path, "w") as f:
    json.dump({"model": MODEL_NAME, "n_layers": n_layers,
               "hidden_size": hidden_size, "seq_len": int(ids.shape[1]),
               "layers": rows}, f, indent=2)
print()
print("Saved:", json_path)

plt.figure(figsize=(11, 5))
plt.plot(range(n_layers), kurts, marker="o")
plt.yscale("log")
plt.xlabel("Layer")
plt.ylabel("Kurtosis (log scale)")
plt.title(f"Activation kurtosis per layer - {tag}")
plt.grid(alpha=0.3)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, f"03_activation_stats_{tag}.png")
plt.savefig(png_path, dpi=150)
print("Saved:", png_path)
