"""
Memory ceiling: given available edge RAM, how many layers can the edge hold?

No GPU and no model weights needed - this reads the model's config (a small
text file) and computes parameter counts from the architecture.

Output: a ceiling FUNCTION (RAM -> deepest feasible split layer), evaluated at
several real-world memory budgets and weight precisions.
"""
import json
import os
from transformers import AutoConfig

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"

# weight precisions the edge might run at, in BYTES per parameter
PRECISIONS = {"bf16": 2.0, "int8": 1.0, "int4": 0.5}

# memory budgets to evaluate (GB). The two odd ones are REAL measurements
# from a 6-year-old handset with 12 GB installed.
BUDGETS_GB = [2, 4, 5.90, 7.81, 12]
REAL_MEASURED = {5.90: "measured: phone under normal use",
                 7.81: "measured: same phone, apps cleared"}

# fraction of free RAM we actually dare to use for model weights.
# The rest is needed for activations, KV cache, the runtime itself, and
# headroom so the OS doesn't kill the process.
USABLE_FRACTION = 0.7


cfg = AutoConfig.from_pretrained(MODEL_NAME)
H = cfg.hidden_size              # 4096
I = cfg.intermediate_size        # 14336
L = cfg.num_hidden_layers        # 32
V = cfg.vocab_size               # 128256
n_heads = cfg.num_attention_heads
n_kv = getattr(cfg, "num_key_value_heads", n_heads)
head_dim = H // n_heads

print(f"Model: {MODEL_NAME}")
print(f"  hidden={H}  intermediate={I}  layers={L}  vocab={V}")
print(f"  attn heads={n_heads}  kv heads={n_kv}  (grouped-query attention)\n")

# ---- parameters in ONE decoder layer ----
# attention: q, k, v, o projections. Llama 3.1 uses grouped-query attention,
# so k and v are SMALLER than q (fewer kv heads).
q = H * H
k = H * (n_kv * head_dim)
v = H * (n_kv * head_dim)
o = H * H
attn = q + k + v + o

# MLP: gate, up, down  (SwiGLU -> three matrices, not two)
mlp = 3 * H * I

# two RMSNorm vectors per layer
norms = 2 * H

per_layer = attn + mlp + norms

# embedding table (edge needs this - it processes the raw input)
embed = V * H

print("Parameters:")
print(f"  attention per layer : {attn:>15,}")
print(f"  mlp per layer       : {mlp:>15,}")
print(f"  norms per layer     : {norms:>15,}")
print(f"  TOTAL per layer     : {per_layer:>15,}")
print(f"  embedding table     : {embed:>15,}")
print(f"  all {L} layers        : {per_layer * L:>15,}")


def ceiling(free_gb, bytes_per_param, usable=USABLE_FRACTION):
    """Deepest split layer k that fits: embedding + k layers of weights."""
    budget_bytes = free_gb * (1024 ** 3) * usable
    embed_bytes = embed * bytes_per_param
    remaining = budget_bytes - embed_bytes
    if remaining < 0:
        return 0, embed_bytes / (1024 ** 3)
    layer_bytes = per_layer * bytes_per_param
    k = int(remaining // layer_bytes)
    return min(k, L)


print(f"\n{'='*70}")
print(f"SPLIT CEILING  (using {int(USABLE_FRACTION*100)}% of free RAM for weights)")
print(f"{'='*70}")
print(f"{'free RAM':>10} " + "".join(f"{p:>10}" for p in PRECISIONS) + "   note")
print("-" * 70)

results = {"model": MODEL_NAME, "params_per_layer": per_layer,
           "embed_params": embed, "usable_fraction": USABLE_FRACTION,
           "ceilings": {}}

for gb in BUDGETS_GB:
    row = f"{gb:>8.2f}GB "
    entry = {}
    for pname, bpp in PRECISIONS.items():
        kk = ceiling(gb, bpp)
        entry[pname] = kk
        row += f"{('L' + str(kk)) if kk < L else 'ALL':>10}"
    note = REAL_MEASURED.get(gb, "")
    print(row + f"   {note}")
    results["ceilings"][str(gb)] = entry

print(f"\n{'='*70}")
print("PER-LAYER AND EMBEDDING SIZE")
print(f"{'='*70}")
for pname, bpp in PRECISIONS.items():
    lay_mb = per_layer * bpp / (1024 ** 2)
    emb_mb = embed * bpp / (1024 ** 2)
    full_gb = (per_layer * L + embed) * bpp / (1024 ** 3)
    print(f"  {pname:>5}: {lay_mb:7.1f} MB per layer | "
          f"{emb_mb:7.1f} MB embedding | {full_gb:5.2f} GB whole model")

print(f"\n{'='*70}")
print("READING THIS")
print(f"{'='*70}")
k_low = ceiling(5.90, 2.0)
k_high = ceiling(7.81, 2.0)
print(f"  At bf16, the SAME phone allows split up to L{k_low} when busy")
print(f"  and up to L{k_high} when cleared. The ceiling MOVES at runtime -")
print(f"  which is exactly why the split decision must be made online, not")
print(f"  offline as EdgeShard does.")
print()
print("  Weight precision changes the ceiling a lot, but does NOT change the")
print("  fact that the activation must still be compressed to cross the")
print("  network - under weight-only quantization the activation is still bf16.")

os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, "04_memory_ceiling.json")
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {path}")
