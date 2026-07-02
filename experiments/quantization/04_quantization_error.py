import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
BIT_WIDTHS = [8, 4, 3]


def quantize_dequantize(x, n_bits):
    # symmetric uniform per-tensor quantization
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1          # e.g. 8-bit -> 127
    scale = absmax / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)   # clamp to valid integer range
    x_hat = q * scale
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

results = []

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

        x = hidden_states.flatten().float()

        # kurtosis (same as before, for comparison)
        mean = x.mean()
        std = x.std()
        normalized = (x - mean) / (std + 1e-8)
        kurtosis = ((normalized ** 4).mean() - 3).item()

        row = {"layer": i, "kurtosis": kurtosis}
        for n_bits in BIT_WIDTHS:
            x_hat = quantize_dequantize(x, n_bits)
            mse = ((x - x_hat) ** 2).mean().item()
            row[f"mse_{n_bits}bit"] = mse

        results.append(row)

        msg = f"Layer {i:02d} | kurtosis: {kurtosis:9.2f}"
        for n_bits in BIT_WIDTHS:
            msg += f" | {n_bits}bit MSE: {row[f'mse_{n_bits}bit']:.6e}"
        print(msg)

os.makedirs("/home/ayush.thakar/thesis/experiments/quantization/results", exist_ok=True)
results_path = "/home/ayush.thakar/thesis/experiments/quantization/results/04_quantization_error.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

layers = [r["layer"] for r in results]
kurt = [r["kurtosis"] for r in results]

fig, ax1 = plt.subplots(figsize=(13, 6))
ax1.set_xlabel("Layer Index")
ax1.set_ylabel("Kurtosis", color="steelblue")
ax1.plot(layers, kurt, color="steelblue", marker="o", label="Kurtosis")
ax1.tick_params(axis="y", labelcolor="steelblue")

ax2 = ax1.twinx()
ax2.set_ylabel("MSE (log scale)")
ax2.set_yscale("log")
for n_bits in BIT_WIDTHS:
    mse_vals = [r[f"mse_{n_bits}bit"] for r in results]
    ax2.plot(layers, mse_vals, marker="s", linestyle="--", label=f"{n_bits}-bit MSE")

ax1.set_title("Kurtosis vs Actual Quantization Error (MSE) per Layer")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
plt.xticks(layers)
plt.tight_layout()
plot_path = "/home/ayush.thakar/thesis/experiments/quantization/results/04_quantization_error.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
