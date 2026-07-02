import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
CHANNEL_OUTLIER_THRESHOLD = 3.0

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

activation_stats = []

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

        flat = hidden_states.flatten().float()

        mean = flat.mean().item()
        std = flat.std().item()
        max_val = flat.abs().max().item()
        min_val = flat.min().item()
        mean_abs = flat.abs().mean().item()
        normalized = (flat - mean) / (std + 1e-8)
        kurtosis = (normalized ** 4).mean().item() - 3
        max_mean_ratio = max_val / (mean_abs + 1e-8)
        min_max_span = flat.max().item() - flat.min().item()

        per_channel_max = hidden_states.abs().max(dim=1).values.flatten().float()
        median_channel_max = per_channel_max.median().item()
        outlier_channel_mask = per_channel_max > (CHANNEL_OUTLIER_THRESHOLD * median_channel_max)
        num_outlier_channels = outlier_channel_mask.sum().item()
        total_channels = per_channel_max.shape[0]
        outlier_channel_pct = 100.0 * num_outlier_channels / total_channels

        sorted_channel_max, _ = torch.sort(per_channel_max, descending=True)
        top_1pct_count = max(1, int(0.01 * total_channels))
        top_1pct_sum = sorted_channel_max[:top_1pct_count].sum().item()
        total_sum = sorted_channel_max.sum().item()
        top_1pct_concentration = 100.0 * top_1pct_sum / (total_sum + 1e-8)

        top_channel_indices = torch.argsort(per_channel_max, descending=True)[:15].tolist()

        stats = {
            "layer": i,
            "mean": mean,
            "std": std,
            "max_abs": max_val,
            "min": min_val,
            "mean_abs": mean_abs,
            "kurtosis": kurtosis,
            "max_mean_ratio": max_mean_ratio,
            "min_max_span": min_max_span,
            "num_outlier_channels": num_outlier_channels,
            "total_channels": total_channels,
            "outlier_channel_pct": outlier_channel_pct,
            "top_1pct_concentration": top_1pct_concentration,
            "top_channel_indices": top_channel_indices,
        }
        activation_stats.append(stats)

        print(f"Layer {i:02d} | kurtosis: {kurtosis:7.2f} | max/mean: {max_mean_ratio:6.2f} | "
              f"span: {min_max_span:6.2f} | outlier channels: {num_outlier_channels:3d}/{total_channels} "
              f"({outlier_channel_pct:5.1f}%) | top-1% mass: {top_1pct_concentration:5.1f}%")

os.makedirs("/home/ayush.thakar/thesis/experiments/quantization/results", exist_ok=True)
results_path = "/home/ayush.thakar/thesis/experiments/quantization/results/03_activation_stats.json"
with open(results_path, "w") as f:
    json.dump(activation_stats, f, indent=2)
print(f"\nResults saved to {results_path}")

layers = [s["layer"] for s in activation_stats]
kurtosis_vals = [s["kurtosis"] for s in activation_stats]
max_mean_vals = [s["max_mean_ratio"] for s in activation_stats]
outlier_pct_vals = [s["outlier_channel_pct"] for s in activation_stats]
concentration_vals = [s["top_1pct_concentration"] for s in activation_stats]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

axes[0, 0].bar(layers, kurtosis_vals, color='steelblue')
axes[0, 0].set_title("Kurtosis per Layer")
axes[0, 0].set_xlabel("Layer Index")
axes[0, 0].set_ylabel("Excess Kurtosis")

axes[0, 1].bar(layers, max_mean_vals, color='indianred')
axes[0, 1].set_title("Max/Mean Ratio per Layer")
axes[0, 1].set_xlabel("Layer Index")
axes[0, 1].set_ylabel("Max/Mean Ratio")

axes[1, 0].bar(layers, outlier_pct_vals, color='seagreen')
axes[1, 0].set_title(f"% Channels Flagged as Outliers (> {CHANNEL_OUTLIER_THRESHOLD}x median)")
axes[1, 0].set_xlabel("Layer Index")
axes[1, 0].set_ylabel("% of 896 Channels")

axes[1, 1].bar(layers, concentration_vals, color='goldenrod')
axes[1, 1].set_title("Top-1% Channels: % of Total Activation Mass")
axes[1, 1].set_xlabel("Layer Index")
axes[1, 1].set_ylabel("% of Total Magnitude")

plt.tight_layout()
plot_path = "/home/ayush.thakar/thesis/experiments/quantization/results/03_activation_stats.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
