import torch
import time
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
NUM_RUNS = 10  # average compute timings to reduce noise
# bandwidths in megabits per second (Mbps) — weak mobile to good wifi
BANDWIDTHS_MBPS = [1, 10, 50, 100]
ACT_DTYPE_BYTES = 2  # bfloat16 activation = 2 bytes per number

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/split_network/results"


def measure_split_compute(model, input_ids, num_runs):
    # returns averaged edge_ms and cloud_ms for every split point,
    # plus the activation size in bytes at the split (constant here)
    n_layers = model.config.num_hidden_layers
    edge_acc = [0.0] * n_layers
    cloud_acc = [0.0] * n_layers
    act_bytes = None

    for _ in range(num_runs):
        for split in range(1, n_layers):
            with torch.no_grad():
                hidden = model.model.embed_tokens(input_ids)
                pos = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
                pos_emb = model.model.rotary_emb(hidden, pos)

                torch.cuda.synchronize()
                t0 = time.time()
                for i in range(split):
                    hidden = model.model.layers[i](
                        hidden, attention_mask=None, position_ids=pos,
                        position_embeddings=pos_emb, past_key_values=None, use_cache=False,
                    )
                torch.cuda.synchronize()
                t1 = time.time()

                if act_bytes is None:
                    act_bytes = hidden.nelement() * ACT_DTYPE_BYTES

                for i in range(split, n_layers):
                    hidden = model.model.layers[i](
                        hidden, attention_mask=None, position_ids=pos,
                        position_embeddings=pos_emb, past_key_values=None, use_cache=False,
                    )
                torch.cuda.synchronize()
                t2 = time.time()

                edge_acc[split] += (t1 - t0) * 1000
                cloud_acc[split] += (t2 - t1) * 1000

    edge_ms = [edge_acc[s] / num_runs for s in range(n_layers)]
    cloud_ms = [cloud_acc[s] / num_runs for s in range(n_layers)]
    return edge_ms, cloud_ms, act_bytes


def transfer_ms(act_bytes, bandwidth_mbps):
    # bytes -> bits, Mbps -> bits per second, result in milliseconds
    bits = act_bytes * 8
    bits_per_sec = bandwidth_mbps * 1_000_000
    seconds = bits / bits_per_sec
    return seconds * 1000


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")

print("Warming up...")
with torch.no_grad():
    h = model.model.embed_tokens(input_ids)
    p = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
    pe = model.model.rotary_emb(h, p)
    for layer in model.model.layers:
        h = layer(h, attention_mask=None, position_ids=p,
                  position_embeddings=pe, past_key_values=None, use_cache=False)
print("Warmup done.\n")

edge_ms, cloud_ms, act_bytes = measure_split_compute(model, input_ids, NUM_RUNS)
print(f"Activation size at split: {act_bytes} bytes ({act_bytes/1024:.2f} KB)\n")

n_layers = model.config.num_hidden_layers
splits = list(range(1, n_layers))

results = {"activation_bytes": act_bytes, "bandwidths": {}}

for bw in BANDWIDTHS_MBPS:
    t_ms = transfer_ms(act_bytes, bw)
    totals = []
    print(f"=== Bandwidth {bw} Mbps (transfer = {t_ms:.4f} ms) ===")
    for s in splits:
        total = edge_ms[s] + t_ms + cloud_ms[s]
        totals.append(total)
        print(f"Split {s:02d} | edge {edge_ms[s]:6.2f} | transfer {t_ms:6.3f} | "
              f"cloud {cloud_ms[s]:6.2f} | TOTAL {total:6.2f} ms")
    best_split = splits[totals.index(min(totals))]
    print(f"--> best split at {bw} Mbps: layer {best_split} (total {min(totals):.2f} ms)\n")
    results["bandwidths"][bw] = {"totals": totals, "best_split": best_split}

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/07_network_split.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")

plt.figure(figsize=(13, 6))
for bw in BANDWIDTHS_MBPS:
    plt.plot(splits, results["bandwidths"][bw]["totals"], marker="o", label=f"{bw} Mbps")
plt.xlabel("Split Point (layer index)")
plt.ylabel("Total latency (ms)")
plt.title("Total Latency vs Split Point, across Network Bandwidths")
plt.legend()
plt.xticks(splits)
plt.tight_layout()
plot_path = f"{RESULTS_DIR}/07_network_split.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
