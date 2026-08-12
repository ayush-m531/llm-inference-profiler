import torch
import json
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- change ONLY this line to switch models ----
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SEQ_LEN = 100
TOP_FRACTION = 0.05     # a channel is "huge" if it is above 5% of the peak
N_GROUPS = 3            # how many quantized groups after the full-precision top
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


def wasted_bits(mags, start, end):
    """How much precision a group loses because one scale must cover all of it.
    start, end are 0-based ranks, end exclusive."""
    if end - start <= 1:
        return 0.0
    hi = mags[start]
    lo = mags[end - 1]
    if lo <= 0:
        return float("inf")
    return math.log2(hi / lo)


def rank_below(mags, threshold):
    """First rank (1-based) whose magnitude falls below threshold."""
    for i, m in enumerate(mags):
        if m < threshold:
            return i + 1
    return len(mags)


def suggest_boundaries(mags):
    """Pick TOP_KEEP by the 5%-of-peak rule, then split what is left
    into N_GROUPS with equal log-magnitude range."""
    peak = mags[0]
    top_keep = rank_below(mags, peak * TOP_FRACTION) - 1
    top_keep = max(1, min(top_keep, len(mags) // 4))

    hi = mags[top_keep]
    lo = max(mags[-1], 1e-9)
    span = math.log10(hi / lo)

    bounds = []
    for g in range(1, N_GROUPS):
        target = hi / (10 ** (span * g / N_GROUPS))
        bounds.append(rank_below(mags, target))
    return top_keep, bounds


print("Loading model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

n_layers = model.config.num_hidden_layers
hidden_size = model.config.hidden_size
print("Layers:", n_layers, " hidden size:", hidden_size)

# spread test layers across the whole depth
TEST_LAYERS = sorted(set([1,
                          n_layers // 4,
                          n_layers // 2,
                          (3 * n_layers) // 4,
                          n_layers - 2]))
print("Testing layers:", TEST_LAYERS)

all_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0]
ids = all_ids[:SEQ_LEN].unsqueeze(0).to("cuda")

print("Running one forward pass...")
with torch.no_grad():
    out = model(ids, output_hidden_states=True)
hs = out.hidden_states

curves = {}
for L in TEST_LAYERS:
    x = hs[L + 1][0].float().cpu()
    per_ch = x.abs().max(dim=0).values
    curves[L] = torch.sort(per_ch, descending=True).values.tolist()
    del x

# ---------- where does the magnitude fall off? ----------
print()
print("WHERE THE MAGNITUDE FALLS OFF")
print("Rank at which a channel drops below each fraction of the peak.")
print()
fracs = [0.5, 0.1, 0.05, 0.01, 0.001]
header = "layer      peak" + "".join(f"{int(f*1000)/10:>9}%" for f in fracs)
print(header)
print("-" * len(header))
for L in TEST_LAYERS:
    m = curves[L]
    row = f"{L:5d} {m[0]:9.1f}"
    for f in fracs:
        row += f"{rank_below(m, m[0] * f):10d}"
    print(row)

# ---------- magnitude at specific ranks ----------
print()
print("MAGNITUDE AT SPECIFIC RANKS")
probe = [r for r in [1, 10, 40, 80, 150, 300, 600, 1000, 2000, hidden_size] if r <= hidden_size]
header = "layer" + "".join(f"{('r' + str(r)):>10}" for r in probe)
print(header)
print("-" * len(header))
for L in TEST_LAYERS:
    m = curves[L]
    print(f"{L:5d}" + "".join(f"{m[r-1]:10.2f}" for r in probe))

# ---------- suggested boundaries ----------
print()
print("SUGGESTED BOUNDARIES PER LAYER")
print("Rule: full precision above 5% of peak, then equal log-range groups.")
print()
print(f"{'layer':>5} {'top_keep':>9} {'bound_1':>9} {'bound_2':>9}   worst wasted bits in a group")
print("-" * 72)
suggestions = {}
for L in TEST_LAYERS:
    m = curves[L]
    tk, bounds = suggest_boundaries(m)
    edges = [tk] + bounds + [len(m)]
    worst = max(wasted_bits(m, edges[i], edges[i + 1]) for i in range(len(edges) - 1))
    suggestions[L] = {"top_keep": tk, "bounds": bounds, "worst_wasted_bits": worst}
    print(f"{L:5d} {tk:9d} {bounds[0]:9d} {bounds[1]:9d}   {worst:8.1f}")

# ---------- compare against the Qwen boundaries ----------
print()
print("COMPARISON: what happens if we just reuse the Qwen numbers (10 / 40 / 150)?")
print("Lower wasted bits is better. inf means a group contains a zero channel.")
print()
print(f"{'layer':>5} {'qwen 10/40/150':>16} {'suggested':>12}")
print("-" * 38)
for L in TEST_LAYERS:
    m = curves[L]
    q_edges = [10, 40, 150, len(m)]
    q_worst = max(wasted_bits(m, q_edges[i], q_edges[i + 1]) for i in range(len(q_edges) - 1))
    print(f"{L:5d} {q_worst:16.1f} {suggestions[L]['worst_wasted_bits']:12.1f}")

# ---------- one boundary set for all layers ----------
tks = [suggestions[L]["top_keep"] for L in TEST_LAYERS]
b1s = [suggestions[L]["bounds"][0] for L in TEST_LAYERS]
b2s = [suggestions[L]["bounds"][1] for L in TEST_LAYERS]
final = (max(tks), max(b1s), max(b2s))

print()
print("=" * 72)
print("USE THESE IN THE NEXT EXPERIMENT")
print("=" * 72)
print(f"  TOP_KEEP = {final[0]}      # channels 1-{final[0]}, full precision")
print(f"  G2_END   = {final[1]}      # channels {final[0]+1}-{final[1]}")
print(f"  G3_END   = {final[2]}      # channels {final[1]+1}-{final[2]}")
print(f"                    # channels {final[2]+1}-{hidden_size}")
print()
print("  Taken as the largest value across layers, so no layer is left with")
print("  huge channels stuck inside a quantized group.")

os.makedirs(OUT_DIR, exist_ok=True)
tag = MODEL_NAME.split("/")[-1]
json_path = os.path.join(OUT_DIR, f"05_02_channel_cliff_{tag}.json")
with open(json_path, "w") as f:
    json.dump({"model": MODEL_NAME, "hidden_size": hidden_size,
               "seq_len": int(ids.shape[1]), "test_layers": TEST_LAYERS,
               "per_layer_suggestion": {str(k): v for k, v in suggestions.items()},
               "final_boundaries": {"TOP_KEEP": final[0], "G2_END": final[1],
                                    "G3_END": final[2]},
               "curves": {str(k): v for k, v in curves.items()}}, f, indent=2)
print()
print("Saved:", json_path)

plt.figure(figsize=(12, 6))
for L in TEST_LAYERS:
    plt.plot(range(1, hidden_size + 1), curves[L], label=f"layer {L}", linewidth=1)
plt.axvline(final[0], color="red", linestyle="--", label=f"top_keep {final[0]}")
plt.axvline(final[1], color="orange", linestyle="--", label=f"G2_END {final[1]}")
plt.axvline(final[2], color="green", linestyle="--", label=f"G3_END {final[2]}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Channel rank (log)")
plt.ylabel("Channel max |activation| (log)")
plt.title(f"Sorted channel magnitudes - {tag}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, f"05_02_channel_cliff_{tag}.png")
plt.savefig(png_path, dpi=150)
print("Saved:", png_path)
