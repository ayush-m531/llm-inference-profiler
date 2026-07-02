import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
NUM_RUNS = 3  # average over multiple runs for stable numbers

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

# ── Measure per-layer latency ────────────────────────────
inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")

layer_latencies = [[] for _ in range(model.config.num_hidden_layers)]

# Warmup — run once without measuring
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

print("Warmup done. Now measuring...")

for run in range(NUM_RUNS):
    print(f"Run {run + 1}/{NUM_RUNS}...")
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        for i, layer in enumerate(model.model.layers):
            start = time.time()
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )
            end = time.time()
            layer_latencies[i].append((end - start) * 1000)

# ── Average across runs ──────────────────────────────────
avg_latencies = [sum(runs) / len(runs) for runs in layer_latencies]

for i, lat in enumerate(avg_latencies):
    print(f"Layer {i:02d} | Avg Latency: {lat:.2f} ms")

print(f"\nTotal inference latency: {sum(avg_latencies):.2f} ms")
print(f"Average per-layer: {sum(avg_latencies)/len(avg_latencies):.2f} ms")

# ── Plot ─────────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.bar(range(len(avg_latencies)), avg_latencies, color='steelblue')
plt.xlabel("Layer Index")
plt.ylabel("Latency (ms)")
plt.title(f"Qwen2.5-0.5B — Per-Layer Latency on CPU (avg over {NUM_RUNS} runs)")
plt.xticks(range(len(avg_latencies)))
plt.axhline(
    y=sum(avg_latencies) / len(avg_latencies),
    color='red', linestyle='--',
    label=f'Average: {sum(avg_latencies)/len(avg_latencies):.1f} ms'
)
plt.legend()
plt.tight_layout()
plt.savefig("/home/ayush.thakar/thesis/experiments/layer_latency.png", dpi=150)
plt.show()
print("Plot saved to experiments/layer_latency.png")
