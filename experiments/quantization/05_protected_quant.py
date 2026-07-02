import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
N_BITS = 4
PROTECT_K = 8  # number of top outlier channels to keep in full precision


def quantize_dequantize(x, n_bits):
    # symmetric uniform per-tensor quantization
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)
    return q * scale


def quantize_protected(hidden, n_bits, protect_k):
    # hidden shape: [1, seq_len, hidden_dim]
    # find top-k channels by max abs value across tokens
    per_channel_max = hidden.abs().max(dim=1).values.flatten()  # [hidden_dim]
    protect_idx = torch.argsort(per_channel_max, descending=True)[:protect_k]

    # mask: which elements belong to protected channels
    hidden_dim = hidden.shape[-1]
    protect_mask = torch.zeros(hidden_dim, dtype=torch.bool, device=hidden.device)
    protect_mask[protect_idx] = True

    # quantize everything, then restore protected channels to original
    x = hidden.float()
    x_hat = quantize_dequantize(x.flatten(), n_bits).reshape(x.shape)
    # restore protected channels (full precision)
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

        x = hidden_states.float()
        x_flat = x.flatten()

        # naive: quantize everything
        x_hat_naive = quantize_dequantize(x_flat, N_BITS)
        mse_naive = ((x_flat - x_hat_naive) ** 2).mean().item()

        # protected: keep top-K channels full precision
        x_hat_prot = quantize_protected(hidden_states, N_BITS, PROTECT_K)
        mse_prot = ((x - x_hat_prot) ** 2).mean().item()

        improvement = mse_naive / (mse_prot + 1e-12)

        results.append({
            "layer": i,
            "mse_naive": mse_naive,
            "mse_protected": mse_prot,
            "improvement_ratio": improvement,
        })

        print(f"Layer {i:02d} | naive MSE: {mse_naive:.4e} | protected MSE: {mse_prot:.4e} | "
              f"improvement: {improvement:8.1f}x")

os.makedirs("/home/ayush.thakar/thesis/experiments/quantization/results", exist_ok=True)
results_path = "/home/ayush.thakar/thesis/experiments/quantization/results/05_protected_quant.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

layers = [r["layer"] for r in results]
naive = [r["mse_naive"] for r in results]
prot = [r["mse_protected"] for r in results]

plt.figure(figsize=(13, 6))
plt.plot(layers, naive, marker="o", label=f"Naive {N_BITS}-bit (all channels)")
plt.plot(layers, prot, marker="s", label=f"Protected {N_BITS}-bit (top-{PROTECT_K} kept full)")
plt.yscale("log")
plt.xlabel("Layer Index")
plt.ylabel("MSE (log scale)")
plt.title(f"Quantization Error: Naive vs Protecting Top-{PROTECT_K} Channels ({N_BITS}-bit)")
plt.legend()
plt.xticks(layers)
plt.tight_layout()
plot_path = "/home/ayush.thakar/thesis/experiments/quantization/results/05_protected_quant.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
