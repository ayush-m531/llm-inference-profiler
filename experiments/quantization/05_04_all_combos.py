import torch
import json
import os
import matplotlib.pyplot as plt
from itertools import product
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
TEST_LAYERS = [3, 10, 20]
TOP_KEEP = 10
# group boundaries: top-10 full | 11-40 | 41-150 | 151-896
G_BOUNDS = [(TOP_KEEP, 40), (40, 150), (150, 896)]
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


def gbytes(nch, nbits, sl):
    return (nch * sl * nbits) / 8 + 4


def fbytes(nch, sl):
    return (nch * sl * 16) / 8


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")
seq_len = input_ids.shape[1]

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


def apply_bits(x, order, bits_combo):
    # bits_combo: (b1, b2, b3) for the three groups
    out = x.clone()
    top = order[:TOP_KEEP]
    out[..., top] = x[..., top]
    nb = fbytes(TOP_KEEP, seq_len)
    for (lo, hi), b in zip(G_BOUNDS, bits_combo):
        idx = order[lo:hi]
        out[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
        nb += gbytes(len(idx), b, seq_len)
    mse = ((x - out) ** 2).mean().item()
    return mse, nb


combos = list(product([4, 8], repeat=3))  # 8 combinations
results = {}
for L in TEST_LAYERS:
    x = captured[L].float()
    order = torch.argsort(x.abs().max(dim=1).values.flatten(), descending=True)
    print(f"\n=== Layer {L} ===  (groups: 11-40 | 41-150 | 151-896)")
    lr = {}
    for combo in combos:
        mse, nb = apply_bits(x, order, combo)
        name = f"{combo[0]}_{combo[1]}_{combo[2]}"
        lr[name] = {"mse": mse, "bytes": nb}
        print(f"  g2={combo[0]} g3={combo[1]} g4={combo[2]} | MSE {mse:.4e} | {nb:8.1f} bytes")
    results[str(L)] = lr

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/05_04_all_combos.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_DIR}/05_04_all_combos.json")

# plot: MSE vs bytes for all 8 combos, layer 10
plt.figure(figsize=(11, 7))
l10 = results["10"]
for name, v in l10.items():
    plt.scatter(v["bytes"], v["mse"], s=70)
    plt.annotate(name, (v["bytes"], v["mse"]), fontsize=8)
plt.yscale("log")
plt.xlabel("Data size (bytes, incl. metadata)")
plt.ylabel("MSE (log)")
plt.title("Layer 10: all 8 precision combos (groups 11-40 | 41-150 | 151-896)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/05_04_all_combos.png", dpi=150)
print("Plot saved.")
