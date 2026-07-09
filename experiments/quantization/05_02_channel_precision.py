import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
TEST_LAYERS = [3, 10, 20]
TOP_KEEP = 10          # channels 1-10: full precision
GROUP2_END = 100       # channels 11-100: group 2
# channels 101-896: group 3

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/quantization/results"


def quant_group(x, n_bits):
    # symmetric per-group quantization; returns dequantized values
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
    # data for a group: (channels * seq_len values * n_bits) / 8  + 4 bytes for the scale
    return (num_channels * seq_len * n_bits) / 8 + 4


def full_bytes(num_channels, seq_len):
    # full precision channels: 16-bit each, no scale needed
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

# warmup + capture activations for the test layers
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

hidden_dim = model.config.hidden_size
seq_len = input_ids.shape[1]


def apply_scheme(hidden, scheme):
    # hidden: [1, seq, dim]. Returns dequantized full tensor + total bytes.
    x = hidden.float().clone()
    dim = x.shape[-1]
    # rank channels by magnitude (max abs across tokens)
    per_channel_max = x.abs().max(dim=1).values.flatten()
    order = torch.argsort(per_channel_max, descending=True)

    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:GROUP2_END]
    g3_idx = order[GROUP2_END:]

    out = x.clone()
    nbytes = 0.0

    if scheme == "uniform_int4":
        out = quant_group(x.flatten(), 4).reshape(x.shape)
        nbytes = group_bytes(dim, 4, seq_len)

    elif scheme == "blob_2tier":
        # top-10 full, rest (11-896) as ONE group at int4
        rest_idx = order[TOP_KEEP:]
        out[..., top_idx] = x[..., top_idx]
        out[..., rest_idx] = quant_group(x[..., rest_idx].flatten(), 4).reshape(
            x[..., rest_idx].shape)
        nbytes = full_bytes(TOP_KEEP, seq_len) + group_bytes(len(rest_idx), 4, seq_len)

    else:
        # 3-tier: scheme is like "g2_4_g3_4" -> g2 bits, g3 bits
        _, g2b, _, g3b = scheme.split("_")
        g2b, g3b = int(g2b), int(g3b)
        out[..., top_idx] = x[..., top_idx]
        out[..., g2_idx] = quant_group(x[..., g2_idx].flatten(), g2b).reshape(
            x[..., g2_idx].shape)
        out[..., g3_idx] = quant_group(x[..., g3_idx].flatten(), g3b).reshape(
            x[..., g3_idx].shape)
        nbytes = (full_bytes(TOP_KEEP, seq_len)
                  + group_bytes(len(g2_idx), g2b, seq_len)
                  + group_bytes(len(g3_idx), g3b, seq_len))

    mse = ((x - out) ** 2).mean().item()
    return mse, nbytes


SCHEMES = ["uniform_int4", "blob_2tier",
           "g2_4_g3_4", "g2_4_g3_8", "g2_8_g3_4", "g2_8_g3_8"]

results = {}
for layer_idx in TEST_LAYERS:
    hidden = captured[layer_idx]
    print(f"\n=== Layer {layer_idx} ===")
    layer_res = {}
    for scheme in SCHEMES:
        mse, nbytes = apply_scheme(hidden, scheme)
        layer_res[scheme] = {"mse": mse, "bytes": nbytes}
        print(f"  {scheme:14s} | MSE {mse:.4e} | {nbytes:8.1f} bytes")
    results[str(layer_idx)] = layer_res

# also dump the sorted channel magnitude for layer 10 (to see the cliff shape)
h10 = captured[10].float()
sorted_mag = torch.sort(h10.abs().max(dim=1).values.flatten(),
                        descending=True).values.tolist()
results["layer10_sorted_channel_magnitude"] = sorted_mag

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/05_02_channel_precision.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

# plot 1: sorted channel magnitude (the cliff) for layer 10
plt.figure(figsize=(12, 5))
plt.plot(range(1, len(sorted_mag) + 1), sorted_mag, color="steelblue")
plt.axvline(TOP_KEEP, color="red", linestyle="--", label=f"top {TOP_KEEP}")
plt.axvline(GROUP2_END, color="orange", linestyle="--", label=f"rank {GROUP2_END}")
plt.yscale("log")
plt.xlabel("Channel rank (by magnitude)")
plt.ylabel("Channel max |activation| (log)")
plt.title("Layer 10: sorted channel magnitudes (where are the group boundaries?)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/05_02_channel_magnitude_cliff.png", dpi=150)

# plot 2: MSE vs bytes for each scheme, layer 10 (the tradeoff)
plt.figure(figsize=(10, 6))
l10 = results["10"]
for scheme in SCHEMES:
    plt.scatter(l10[scheme]["bytes"], l10[scheme]["mse"], s=60)
    plt.annotate(scheme, (l10[scheme]["bytes"], l10[scheme]["mse"]), fontsize=8)
plt.yscale("log")
plt.xlabel("Data size (bytes, incl. metadata)")
plt.ylabel("MSE (log)")
plt.title("Layer 10: error vs data size for each scheme")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/05_02_channel_precision.png", dpi=150)
print("Plots saved.")
