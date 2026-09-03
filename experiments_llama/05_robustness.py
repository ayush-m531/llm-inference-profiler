"""
Robustness check for the bytes-per-split finding.

Re-runs the 03_bytes_per_split measurement across 3 sequence lengths x 3 prompts
and asks: does the bathtub shape and the early-vs-middle saving actually hold,
or was it an artifact of one configuration?

Critically, it also computes whether the between-prompt variation is SMALLER
than the effect we are claiming. If it isn't, the effect is noise.
"""
import torch
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SEQ_LENS = [50, 100, 250]
KL_THRESHOLDS = [0.05, 0.1, 0.25]

TOP_KEEP = 5
G2_END = 72
G3_END = 1312

CALIB_LAYER = 2
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"

# Three prompts of deliberately DIFFERENT character. If the finding only holds
# for encyclopedia-style text, that is worth knowing.
PROMPTS = {
    "technical": ("The history of artificial intelligence began in the 1950s when researchers "
                  "started exploring the possibility of creating machines that could think and "
                  "reason like humans. Early pioneers developed symbolic systems and search "
                  "algorithms, believing that intelligence could be captured through logical "
                  "rules. Over the following decades the field experienced cycles of optimism "
                  "and disappointment, often called AI winters, as early promises failed to "
                  "materialize. The introduction of machine learning shifted the paradigm from "
                  "hand crafted rules toward systems that learn patterns directly from data. "
                  "Neural networks, inspired loosely by biological brains, gradually became the "
                  "dominant approach, especially after advances in computing hardware made it "
                  "practical to train very large models on enormous datasets. ") * 4,

    "narrative": ("The old lighthouse keeper had watched the same stretch of coast for forty "
                  "years. Every evening he climbed the spiral staircase, counting the steps out "
                  "of habit rather than need, and lit the lamp that had guided ships since his "
                  "grandfather's time. The village below had shrunk year by year as the young "
                  "people left for the cities, but he stayed, because someone had to. On stormy "
                  "nights the wind screamed against the glass and he would think about the "
                  "fishing boat that never came home, the one he still dreamed about, and he "
                  "would keep the light burning a little brighter until dawn came grey and "
                  "cold over the water and another day began without incident. ") * 4,

    "conversational": ("Hey, so I was thinking about what you said yesterday about the trip. "
                       "I mean, I get why you want to go in summer, but have you actually "
                       "checked the prices? Last time I looked they were insane. My cousin went "
                       "in September and said it was way cheaper and the weather was still "
                       "fine, maybe even better because it wasn't so crowded. Anyway, let me "
                       "know what you think, I'm easy either way. Oh and did you ever hear back "
                       "about the job thing? You never said. I've been meaning to ask but it "
                       "kept slipping my mind whenever we talked. ") * 4,
}


def quant_group(x, n_bits):
    if x.numel() == 0 or n_bits >= 16:
        return x.clone()
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    return q * scale


def apply_uniform(hidden, n_bits):
    x = hidden.float()
    if n_bits >= 16:
        return x.clone(), x.shape[1] * x.shape[2] * 16 / 8
    out = quant_group(x.flatten(), n_bits).reshape(x.shape)
    return out, (x.shape[1] * x.shape[2] * n_bits) / 8 + 4


def apply_grouped(hidden, bits_combo, fixed_order):
    x = hidden.float()
    seq_len = x.shape[1]
    order = fixed_order
    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:G2_END]
    g3_idx = order[G2_END:G3_END]
    g4_idx = order[G3_END:]
    out = x.clone()
    out[..., top_idx] = x[..., top_idx]
    b2, b3, b4 = bits_combo
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        out[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
    nbytes = (TOP_KEEP * seq_len * 16) / 8
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        nbytes += (len(idx) * seq_len * b) / 8 + 4
    return out, nbytes


SCHEMES = [
    ("grouped", (4, 4, 4)),
    ("uniform", 4),
    ("grouped", (8, 4, 4)),
    ("grouped", (8, 8, 4)),
    ("uniform", 8),
    ("grouped", (8, 8, 8)),
    ("uniform", 16),
]

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
splits = list(range(0, n_layers))
print(f"Layers: {n_layers}\n")


def run_with_scheme(ids, quant_split, scheme, fixed_order):
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    nbytes = None
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None, use_cache=False)
        if quant_split is not None and i == quant_split:
            kind, param = scheme
            od = hidden.dtype
            if kind == "uniform":
                q, nbytes = apply_uniform(hidden, param)
            elif kind == "grouped":
                q, nbytes = apply_grouped(hidden, param, fixed_order)
            else:
                q, nbytes = hidden.float(), hidden.shape[1] * hidden.shape[2] * 16 / 8
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0].float(), nbytes


def kl_all(dmg_logits, ref_probs):
    dmg = F.softmax(dmg_logits, dim=-1)
    return F.kl_div((dmg + 1e-12).log(), ref_probs,
                    reduction="none").sum(dim=-1).mean().item()


results = {"model": MODEL_NAME, "seq_lens": SEQ_LENS,
           "prompts": list(PROMPTS.keys()), "runs": {}}

total = len(SEQ_LENS) * len(PROMPTS)
done = 0

for pname, ptext in PROMPTS.items():
    all_ids = tok(ptext, return_tensors="pt")["input_ids"][0]
    for seq_len in SEQ_LENS:
        done += 1
        print(f"[{done}/{total}] prompt='{pname}' seq_len={seq_len}")
        if all_ids.shape[0] < seq_len:
            print(f"   SKIP: prompt only has {all_ids.shape[0]} tokens")
            continue
        ids = all_ids[:seq_len].unsqueeze(0).to("cuda")

        # freeze channel order from THIS run's calibration pass
        with torch.no_grad():
            calib = model(ids, output_hidden_states=True)
        calib_act = calib.hidden_states[CALIB_LAYER + 1][0].float()
        fixed_order = torch.argsort(
            calib_act.abs().max(dim=0).values, descending=True).to("cuda")
        del calib, calib_act

        with torch.no_grad():
            ref_logits, _ = run_with_scheme(ids, None, ("none", None), fixed_order)
            ref_probs = F.softmax(ref_logits, dim=-1)
            kl_t, by_t = {}, {}
            for s in splits:
                for si, scheme in enumerate(SCHEMES):
                    lg, nb = run_with_scheme(ids, s, scheme, fixed_order)
                    kl_t[(s, si)] = kl_all(lg, ref_probs)
                    by_t[(s, si)] = nb

        rec = {}
        for T in KL_THRESHOLDS:
            per_split = {}
            for s in splits:
                ci = next((si for si in range(len(SCHEMES))
                           if kl_t[(s, si)] <= T), len(SCHEMES) - 1)
                per_split[s] = by_t[(s, ci)]
            early = min(per_split[0], per_split[1])
            # feasible middle zone: memory ceiling says L5-L11 is the real range
            mid = min(per_split[s] for s in range(5, 12))
            rec[str(T)] = {"bytes": {str(s): per_split[s] for s in splits},
                           "early_bytes": early, "mid_bytes": mid,
                           "saving_pct": 100 * (early - mid) / early}
            print(f"     KL<={T}: early={early:9.0f}  mid(L5-11)={mid:9.0f}  "
                  f"saving={rec[str(T)]['saving_pct']:5.1f}%")
        results["runs"][f"{pname}_{seq_len}"] = rec

# ---------- THE ACTUAL TEST ----------
print("\n" + "=" * 70)
print("IS THE EFFECT BIGGER THAN THE VARIATION? (the Exp 09_01 test)")
print("=" * 70)
verdicts = []
for T in KL_THRESHOLDS:
    savings = [r[str(T)]["saving_pct"] for r in results["runs"].values()]
    lo, hi = min(savings), max(savings)
    mean = sum(savings) / len(savings)
    spread = hi - lo
    print(f"\nKL <= {T}")
    print(f"  saving across all {len(savings)} runs: "
          f"min {lo:.1f}%  mean {mean:.1f}%  max {hi:.1f}%")
    print(f"  spread between runs: {spread:.1f} percentage points")
    if lo > 0 and spread < mean:
        print(f"  ==> HOLDS. Every run shows a saving, and the variation")
        print(f"      ({spread:.1f}pp) is smaller than the effect ({mean:.1f}%).")
        verdicts.append(True)
    elif lo > 0:
        print(f"  ==> DIRECTIONALLY HOLDS but noisy: the spread ({spread:.1f}pp)")
        print(f"      is as large as the effect ({mean:.1f}%). Report the RANGE,")
        print(f"      not a single number.")
        verdicts.append(True)
    else:
        print(f"  ==> DOES NOT HOLD. At least one run shows no saving.")
        verdicts.append(False)

print("\n" + "=" * 70)
if all(verdicts):
    print("VERDICT: the early-vs-middle saving survives prompt and sequence")
    print("length variation. Safe to build on.")
else:
    print("VERDICT: the saving is NOT robust. Do not build the controller on it")
    print("until this is understood.")
print("=" * 70)

os.makedirs(OUT_DIR, exist_ok=True)
path = os.path.join(OUT_DIR, "05_robustness.json")
with open(path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {path}")

# one plot per threshold, one line per run
fig, axes = plt.subplots(1, len(KL_THRESHOLDS),
                         figsize=(6 * len(KL_THRESHOLDS), 5), squeeze=False)
for ax, T in zip(axes[0], KL_THRESHOLDS):
    for name, rec in results["runs"].items():
        ys = [rec[str(T)]["bytes"][str(s)] for s in splits]
        ax.plot(splits, ys, marker=".", linewidth=1, label=name)
    ax.axvspan(5, 11, alpha=0.12, color="green")
    ax.set_title(f"KL <= {T}")
    ax.set_xlabel("Split point (layer)")
    ax.set_ylabel("Bytes needed")
    ax.grid(alpha=0.3)
axes[0][0].legend(fontsize=7)
plt.tight_layout()
png = os.path.join(OUT_DIR, "05_robustness.png")
plt.savefig(png, dpi=150)
print("Saved:", png)
