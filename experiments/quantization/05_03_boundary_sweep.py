import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
TEST_LAYERS = [3, 10, 20]
TOP_KEEP = 10
GROUP2_ENDS = [20, 30, 40, 50, 75, 100, 150]  # sweep the group-2 boundary
G2_BITS = 8   # winning scheme from 05_02: middle group int8
G3_BITS = 4   # bulk int4

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/quantization/results"


def quant_group(x, n_bits):
    if x.numel() == 0:
        return x.clone()
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def group_bytes(num_channels, n_bits, seq_len):
    return (num_channels * seq_len * n_bits) / 8 + 4


def full_bytes(num_channels, seq_len):
    return (num_channels * seq_len * 16) / 8


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

captured = {}
with torch.no_grad():
    hidden = model.model.embed_tokens(input_ids)
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None, use_cache=False)
        if i in TEST_LAYERS:
            captured[i] = hidden.clone()

seq_len = input_ids.shape[1]


def apply_scheme(hidden, g2_end):
    x = hidden.float().clone()
    per_channel_max = x.abs().max(dim=1).values.flatten()
    order = torch.argsort(per_channel_max, descending=True)

    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:g2_end]
    g3_idx = order[g2_end:]

    out = x.clone()
    out[..., top_idx] = x[..., top_idx]
    out[..., g2_idx] = quant_group(x[..., g2_idx].flatten(), G2_BITS).reshape(
        x[..., g2_idx].shape)
    out[..., g3_idx] = quant_group(x[..., g3_idx].flatten(), G3_BITS).reshape(
        x[..., g3_idx].shape)

    nbytes = (full_bytes(TOP_KEEP, seq_len)
              + group_bytes(len(g2_idx), G2_BITS, seq_len)
              + group_bytes(len(g3_idx), G3_BITS, seq_len))
    mse = ((x - out) ** 2).mean().item()
    return mse, nbytes


results = {}
for layer_idx in TEST_LAYERS:
    hidden = captured[layer_idx]
    print(f"\n=== Layer {layer_idx} ===")
    layer_res = {}
    for g2_end in GROUP2_ENDS:
        mse, nbytes = apply_scheme(hidden, g2_end)
        layer_res[str(g2_end)] = {"mse": mse, "bytes": nbytes}
        print(f"  g2_end={g2_end:3d} ({g2_end-TOP_KEEP:3d} chans in int8) | "
              f"MSE {mse:.4e} | {nbytes:8.1f} bytes")
    results[str(layer_idx)] = layer_res

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/05_03_boundary_sweep.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

# plot: MSE vs group-2 boundary, one line per layer
plt.figure(figsize=(11, 6))
for layer_idx in TEST_LAYERS:
    mses = [results[str(layer_idx)][str(g)]["mse"] for g in GROUP2_ENDS]
    plt.plot(GROUP2_ENDS, mses, marker="o", label=f"Layer {layer_idx}")
plt.yscale("log")
plt.xlabel("Group-2 end boundary (channels 11..end at int8)")
plt.ylabel("MSE (log)")
plt.title("Effect of group-2 boundary on error (top-10 full, g2=int8, g3=int4)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/05_03_boundary_sweep.png", dpi=150)
print("Plot saved.")
