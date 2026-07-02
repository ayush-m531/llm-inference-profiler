import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"

# ── Load model ──────────────────────────────────────────
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

# ── Warmup ────────────────────────────────────────────────
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

# ── Split Point Analysis ──────────────────────────────────
split_results = []

for split in range(1, model.config.num_hidden_layers):
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        # Edge
        edge_start = time.time()
        for i in range(split):
            hidden_states = model.model.layers[i](
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )
        edge_end = time.time()

        activation_size_kb = hidden_states.nelement() * hidden_states.element_size() / 1024

        # Cloud
        cloud_start = time.time()
        for i in range(split, model.config.num_hidden_layers):
            hidden_states = model.model.layers[i](
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )
        cloud_end = time.time()

    edge_ms = (edge_end - edge_start) * 1000
    cloud_ms = (cloud_end - cloud_start) * 1000

    split_results.append({
        "split": split,
        "edge_ms": edge_ms,
        "cloud_ms": cloud_ms,
        "total_ms": edge_ms + cloud_ms,
        "activation_kb": activation_size_kb
    })
    print(f"Split {split:02d} | Edge: {edge_ms:.2f}ms | Cloud: {cloud_ms:.2f}ms | Total: {edge_ms+cloud_ms:.2f}ms")

# ── Find best split under sequential model (total latency) ──
best = min(split_results, key=lambda r: r["total_ms"])
print(f"\nBest split point (lowest total latency): {best['split']} (Total: {best['total_ms']:.2f}ms)")

# ── Plot ──────────────────────────────────────────────────
splits = [r["split"] for r in split_results]
edge_ms_list = [r["edge_ms"] for r in split_results]
cloud_ms_list = [r["cloud_ms"] for r in split_results]
total_ms_list = [r["total_ms"] for r in split_results]

plt.figure(figsize=(12, 5))
plt.plot(splits, edge_ms_list, label="Edge latency", marker='o')
plt.plot(splits, cloud_ms_list, label="Cloud latency", marker='o')
plt.plot(splits, total_ms_list, label="Total latency (sequential)", marker='o', linestyle='--', color='black')
plt.xlabel("Split Point (layer index)")
plt.ylabel("Latency (ms)")
plt.title("Edge vs Cloud Latency at Different Split Points (GPU)")
plt.legend()
plt.xticks(splits)
plt.tight_layout()
plt.savefig("/home/ayush.thakar/thesis/experiments/split_analysis.png", dpi=150)
print("Plot saved to /home/ayush.thakar/thesis/experiments/split_analysis.png")
