"""
Memory ceiling: given available edge RAM, what is the deepest split point?

No GPU and no model weights needed - reads the model's config and computes
parameter counts from the architecture.

CONVENTION: split index k means the edge computes layers 0..k INCLUSIVE, i.e.
k+1 layers. So if k layers fit in memory, the deepest split index is k-1.
An earlier version printed the layer COUNT as if it were an INDEX, making every
ceiling one layer too deep.

Output: a ceiling FUNCTION (RAM -> deepest feasible split index), evaluated at
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
# The rest covers activations, KV cache, the runtime, and OS headroom.
# THIS IS A JUDGEMENT CALL, NOT A MEASUREMENT. State it in the thesis.
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
k_proj = H * (n_kv * head_dim)
v_proj = H * (n_kv * head_dim)
o = H * H
attn = q + k_proj + v_proj + o

# MLP: gate, up, down  (SwiGLU -> three matrices, not two)
mlp = 3 * H * I

# two RMSNorm vectors per layer
norms = 2 * H

per_layer = attn + mlp + norms

# embedding table (edge needs this once - it processes the raw input)
embed = V * H

print("Parameters:")
print(f"  attention per layer : {attn:>15,}")
print(f"  mlp per layer       : {mlp:>15,}")
print(f"  norms per layer     : {norms:>15,}")
print(f"  TOTAL per layer     : {per_layer:>15,}")
print(f"  embedding table     : {embed:>15,}")
print(f"  all {L} layers        : {per_layer * L:>15,}")


def layers_that_fit(free_gb, bytes_per_param, usable=USABLE_FRACTION):
    """How many whole layers fit, after the embedding is accounted for."""
    budget_bytes = free_gb * (1024 ** 3) * usable
    remaining = budget_bytes - embed * bytes_per_param
    if remaining < 0:
        return 0
    return int(remaining // (per_layer * bytes_per_param))


def ceiling(free_gb, bytes_per_param, usable=USABLE_FRACTION):
    """Deepest split INDEX that fits. Returns -1 if not even one layer fits.

    k layers fitting means layers 0..k-1 fit, so the deepest split index is
    k-1 (because split k means layers 0..k inclusive = k+1 layers).
    """
    k = layers_that_fit(free_gb, bytes_per_param, usable)
    if k <= 0:
        return -1
    return min(k - 1, L - 1)


def fmt(idx):
    if idx < 0:
        return "none"
    if idx >= L - 1:
        return "ALL"
    return "L" + str(idx)


print(f"\n{'='*72}")
print(f"DEEPEST FEASIBLE SPLIT INDEX  "
      f"({int(USABLE_FRACTION*100)}% of free RAM used for weights)")
print(f"{'='*72}")
print(f"{'free RAM':>10} " + "".join(f"{p:>10}" for p in PRECISIONS) + "   note")
print("-" * 72)

results = {"model": MODEL_NAME, "params_per_layer": per_layer,
           "embed_params": embed, "usable_fraction": USABLE_FRACTION,
           "convention": "split k = layers 0..k inclusive = k+1 layers",
           "ceilings": {}, "layers_fit": {}}

for gb in BUDGETS_GB:
    row = f"{gb:>8.2f}GB "
    entry, fits = {}, {}
    for pname, bpp in PRECISIONS.items():
        idx = ceiling(gb, bpp)
        entry[pname] = idx
        fits[pname] = layers_that_fit(gb, bpp)
        row += f"{fmt(idx):>10}"
    print(row + f"   {REAL_MEASURED.get(gb, '')}")
    results["ceilings"][str(gb)] = entry
    results["layers_fit"][str(gb)] = fits

print(f"\n{'='*72}")
print("WORKING (bf16) - showing the count-to-index step explicitly")
print(f"{'='*72}")
for gb in BUDGETS_GB:
    budget = gb * (1024 ** 3) * USABLE_FRACTION
    rem = budget - embed * 2
    exact = rem / (per_layer * 2)
    n = layers_that_fit(gb, 2.0)
    idx = ceiling(gb, 2.0)
    print(f"  {gb:>5.2f} GB -> {exact:6.3f} layers fit -> {n:2d} whole layers "
          f"(L0..L{n-1}) -> deepest split {fmt(idx)}")

print(f"\n{'='*72}")
print("PER-LAYER AND EMBEDDING SIZE")
print(f"{'='*72}")
for pname, bpp in PRECISIONS.items():
    lay_mb = per_layer * bpp / (1024 ** 2)
    emb_mb = embed * bpp / (1024 ** 2)
    full_gb = (per_layer * L + embed) * bpp / (1024 ** 3)
    print(f"  {pname:>5}: {lay_mb:7.1f} MB per layer | "
          f"{emb_mb:7.1f} MB embedding | {full_gb:5.2f} GB whole model")

print(f"\n{'='*72}")
print("READING THIS")
print(f"{'='*72}")
k_low = ceiling(5.90, 2.0)
k_high = ceiling(7.81, 2.0)
print(f"  At bf16 the SAME phone allows split up to {fmt(k_low)} when busy")
print(f"  and up to {fmt(k_high)} when cleared. The ceiling MOVES at runtime,")
print(f"  which is why the split decision must be made online rather than")
print(f"  offline as EdgeShard does.")
print()
margin = (7.81 * (1024**3) * USABLE_FRACTION - embed * 2) / (per_layer * 2)
print(f"  KNIFE EDGE: 7.81 GB gives {margin:.3f} layers - only a "
      f"{100*(margin - int(margin))/int(margin):.1f}% margin.")
for uf in [0.65, 0.67, 0.70, 0.75]:
    print(f"    USABLE_FRACTION {uf:.2f} -> deepest split "
          f"{fmt(ceiling(7.81, 2.0, uf))}")
print()
print("  Weight precision changes the ceiling a lot but does NOT change the")
print("  fact that the activation must still be compressed to cross the")
print("  network - under weight-only quantization the activation is still bf16.")

os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, "04_memory_ceiling.json")
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {path}")
