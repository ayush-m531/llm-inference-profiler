import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
N_BITS = 4
K_VALUES = [0, 2, 4, 8, 16, 32, 64, 128]
TEST_LAYERS = [1, 3, 10, 20]


def quantize_dequantize(x, n_bits):
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)
    return q * scale


def quantize_protected(hidden, n_bits, protect_k):
    x = hidden.float()
    if protect_k == 0:
        x_hat = quantize_dequantize(x.flatten(), n_bits).reshape(x.shape)
        return x_hat
    per_channel_max = hidden.abs().max(dim=1).values.flatten()
    protect_idx = torch.argsort(per_channel_max, descending=True)[:protect_k]
    hidden_dim = hidden.shape[-1]
    protect_mask = torch.zeros(hidden_dim, dtype=torch.bool, device=hidden.device)
    protect_mask[protect_idx] = True
    x_hat = quantize_dequantize(x.flatten(), n_bits).reshape(x.shape)
    x_hat[..., protect_mask] = x[..., protect_mask]
    return x_hat


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model.eval()
print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")

print("Warming up...")
with torch.no_grad():
    warmup_hidden = model.model.embed_tokens(input_ids)
    warmup_pos = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
    warmup_embeddings = model.model.rotary_emb(warmup_hidden, warmup_pos)
    for layer in model.model.layers:
        warmup_hidden = layer(
            warmup_hidden,
            attention_mask=None,
            position_ids=warmup_pos,
            position_embeddings=warmup_embeddings,
            past_key_values=None,
            use_cache=False,
        )
print("Warmup done.\n")

# capture each test layer's activation
captured = {}
with torch.no_grad():
    hidden_states = model.model.embed_tokens(input_ids)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    for i, layer in enumerate(model.model.layers):
        hidden_states = layer(
            hidden_states,
            attention_mask=None,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            past_key_values=None,
            use_cache=False,
        )
        if i in TEST_LAYERS:
            captured[i] = hidden_states.clone()

hidden_dim = model.config.hidden_size
results = {}

for layer_idx in TEST_LAYERS:
    hidden = captured[layer_idx]
    x = hidden.float()
    layer_curve = []
    print(f"\n=== Layer {layer_idx} ===")
    for k in K_VALUES:
        x_hat = quantize_protected(hidden, N_BITS, k)
        mse = ((x - x_hat) ** 2).mean().item()
        pct = 100.0 * k / hidden_dim
        layer_curve.append({"k": k, "mse": mse, "pct_channels": pct})
        print(f"K={k:3d} ({pct:4.1f}% channels protected) | MSE: {mse:.4e}")
    results[layer_idx] = layer_curve

os.makedirs("/home/ayush.thakar/thesis/experiments/quantization/results", exist_ok=True)
results_path = "/home/ayush.thakar/thesis/experiments/quantization/results/05_01_k_sweep.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

plt.figure(figsize=(12, 6))
for layer_idx in TEST_LAYERS:
    ks = [pt["k"] for pt in results[layer_idx]]
    mses = [pt["mse"] for pt in results[layer_idx]]
    plt.plot(ks, mses, marker="o", label=f"Layer {layer_idx}")
plt.yscale("log")
plt.xlabel("K (channels kept in full precision)")
plt.ylabel("MSE (log scale)")
plt.title(f"Quantization Error vs Protected Channels ({N_BITS}-bit)")
plt.legend()
plt.tight_layout()
plot_path = "/home/ayush.thakar/thesis/experiments/quantization/results/05_01_k_sweep.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
