import torch
import time
import json
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
# 8 varied prompts, similar length (~5-8 tokens), different topics/styles
PROMPTS = [
    "Hello, my name is",
    "The weather today is very",
    "In the year 2050, humans will",
    "def add(a, b): return",
    "The capital of France is",
    "She opened the door and saw",
    "Breaking news: scientists have discovered",
    "My favorite food to eat is",
]
NUM_TIMING_RUNS = 10
BIT_WIDTHS = [16, 8, 4, 3]
SCENARIOS = [(1, 1), (1, 50), (100, 1), (100, 50)]
KL_THRESHOLDS = [0.5, 1.0, 2.0]

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/split_network/results"


def quantize_dequantize(x, n_bits):
    if n_bits >= 16:
        return x
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)
    return q * scale


def data_bytes(num_elements, n_bits):
    return (num_elements * n_bits) / 8 + 4


def transfer_ms(nbytes, bandwidth_mbps):
    return ((nbytes * 8) / (bandwidth_mbps * 1_000_000)) * 1000


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
)
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. Layers: {n_layers}\n")


def run_to_logits(input_ids, quant_split=None, n_bits=16):
    hidden = model.model.embed_tokens(input_ids)
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None, use_cache=False)
        if quant_split is not None and i == quant_split:
            od = hidden.dtype
            hidden = quantize_dequantize(hidden.float(), n_bits).to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0, -1, :].float()


# --- measure compute times ONCE (timing is prompt-length dependent but our prompts
#     are similar length; we average over all prompts for a representative timing) ---
print("Measuring compute times (averaged over prompts)...")
edge_ms = [0.0] * n_layers
cloud_ms = [0.0] * n_layers
act_elems = None
count = 0
with torch.no_grad():
    # warmup
    wu = tokenizer(PROMPTS[0], return_tensors="pt")["input_ids"].to("cuda")
    _ = run_to_logits(wu)
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
        for _ in range(NUM_TIMING_RUNS):
            for split in range(1, n_layers):
                hidden = model.model.embed_tokens(ids)
                pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
                pos_emb = model.model.rotary_emb(hidden, pos)
                torch.cuda.synchronize(); t0 = time.time()
                for i in range(split):
                    hidden = model.model.layers[i](hidden, attention_mask=None, position_ids=pos,
                        position_embeddings=pos_emb, past_key_values=None, use_cache=False)
                torch.cuda.synchronize(); t1 = time.time()
                for i in range(split, n_layers):
                    hidden = model.model.layers[i](hidden, attention_mask=None, position_ids=pos,
                        position_embeddings=pos_emb, past_key_values=None, use_cache=False)
                torch.cuda.synchronize(); t2 = time.time()
                edge_ms[split] += (t1 - t0) * 1000
                cloud_ms[split] += (t2 - t1) * 1000
        count += 1
# average over (prompts x runs)
denom = count * NUM_TIMING_RUNS
edge_ms = [e / denom for e in edge_ms]
cloud_ms = [c / denom for c in cloud_ms]

# use a representative activation element count (from a mid-length prompt)
mid_ids = tokenizer(PROMPTS[0], return_tensors="pt")["input_ids"].to("cuda")
with torch.no_grad():
    h = model.model.embed_tokens(mid_ids)
    act_elems = h.nelement()  # elements at split (constant across layers)

# --- measure KL per (prompt, split, bits) ---
print("Measuring quality per prompt...\n")
# kl_per_prompt[prompt_idx][(split, bits)] = kl
kl_per_prompt = []
with torch.no_grad():
    for pidx, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
        ref = F.softmax(run_to_logits(ids, None, 16), dim=-1)
        table = {}
        for split in range(1, n_layers):
            for b in BIT_WIDTHS:
                if b >= 16:
                    table[(split, b)] = 0.0
                    continue
                dmg = F.softmax(run_to_logits(ids, split, b), dim=-1)
                table[(split, b)] = F.kl_div((dmg + 1e-12).log(), ref, reduction="sum").item()
        kl_per_prompt.append(table)
        print(f"  prompt {pidx}: '{prompt}' done")

# --- controller decision per prompt per scenario ---
splits = list(range(1, n_layers))
results = {"prompts": PROMPTS, "scenarios": []}

print("\n=== Controller decision stability across prompts ===")
for bw, slowdown in SCENARIOS:
    scenario_rec = {"bandwidth_mbps": bw, "slowdown": slowdown, "threshold_picks": {}}
    print(f"\n--- {bw} Mbps, slowdown x{slowdown} ---")
    for T in KL_THRESHOLDS:
        picks = []
        for pidx in range(len(PROMPTS)):
            options = []
            for s in splits:
                for b in BIT_WIDTHS:
                    nbytes = data_bytes(act_elems, b)
                    total = edge_ms[s] * slowdown + transfer_ms(nbytes, bw) + cloud_ms[s]
                    kl = kl_per_prompt[pidx][(s, b)]
                    if kl <= T:
                        options.append((total, s, b, kl))
            if options:
                total, s, b, kl = min(options, key=lambda o: o[0])
                picks.append({"prompt": pidx, "split": s, "bits": b})
            else:
                picks.append({"prompt": pidx, "split": None, "bits": None})
        # how consistent are the picks?
        pick_set = set((p["split"], p["bits"]) for p in picks)
        scenario_rec["threshold_picks"][str(T)] = {"picks": picks,
                                                   "unique_choices": len(pick_set)}
        summary = {}
        for p in picks:
            key = f"split {p['split']}, {p['bits']}-bit"
            summary[key] = summary.get(key, 0) + 1
        agree = max(summary.values())
        print(f"  KL<={T}: {agree}/{len(PROMPTS)} prompts agree on same choice | "
              f"{len(pick_set)} unique choice(s): {dict(summary)}")
    results["scenarios"].append(scenario_rec)

os.makedirs(RESULTS_DIR, exist_ok=True)
results_path = f"{RESULTS_DIR}/08_02_multiprompt_controller.json"
with open(results_path, "w") as f:
    # convert kl tables to serializable form
    serial_kl = []
    for table in kl_per_prompt:
        serial_kl.append({f"{s}_{b}": v for (s, b), v in table.items()})
    json.dump({"results": results, "kl_per_prompt": serial_kl}, f, indent=2)
print(f"\nResults saved to {results_path}")

# --- plot: KL vs layer at 4-bit, one line per prompt (structure stability) ---
plt.figure(figsize=(13, 6))
for pidx, prompt in enumerate(PROMPTS):
    kl_4bit = [kl_per_prompt[pidx][(s, 4)] for s in splits]
    plt.plot(splits, kl_4bit, marker="o", markersize=3, alpha=0.7,
             label=f"p{pidx}")
plt.xlabel("Split point (layer)")
plt.ylabel("KL divergence at 4-bit")
plt.title("Per-layer 4-bit quantization damage across 8 prompts (structure stability)")
plt.legend(fontsize=8, ncol=2)
plt.xticks(splits)
plt.tight_layout()
plot_path = f"{RESULTS_DIR}/08_02_multiprompt_controller.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
