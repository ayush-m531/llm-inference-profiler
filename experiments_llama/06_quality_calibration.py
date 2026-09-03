"""
Quality calibration: relate KL divergence to perplexity and to observed output.

Measures, for each compression scheme at three split points:
  1. KL divergence against the uncompressed reference
  2. Perplexity on held-out text
  3. Generated output

Purpose: establish what a given KL budget corresponds to in measurable quality
terms, so budgets are derived rather than assumed.

Writes both a JSON file (machine-readable) and a TXT transcript (readable).
"""
import torch
import json
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "06_quality_calibration"

# L1 = fragile early layer (contrast). L7 = memory ceiling when phone is busy.
# L11 = memory ceiling when phone is cleared.
SPLITS = [1, 7, 11]

TOP_KEEP = 5
G2_END = 72
G3_END = 1312
CALIB_LAYER = 2

PPL_SEQ_LEN = 256
GEN_TOKENS = 40

PPL_TEXT = ("Climate scientists have documented a steady rise in global average "
            "temperatures over the past century, driven primarily by the "
            "accumulation of greenhouse gases in the atmosphere. The consequences "
            "extend well beyond simple warming: shifting rainfall patterns, more "
            "frequent extreme weather events, rising sea levels, and disruptions "
            "to agriculture and ecosystems. Mitigation strategies focus on "
            "reducing emissions through renewable energy adoption, improved "
            "efficiency, and changes in land use, while adaptation measures aim "
            "to reduce vulnerability to changes already underway. The pace of "
            "both remains a subject of intense policy debate across nations, "
            "with developing countries arguing that historical responsibility "
            "should shape the distribution of costs and obligations. ") * 3

GEN_PROMPT = "The three most important things to know about machine learning are"


_lines = []


def out(s=""):
    print(s)
    _lines.append(s)


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
    o = quant_group(x.flatten(), n_bits).reshape(x.shape)
    return o, (x.shape[1] * x.shape[2] * n_bits) / 8 + 4


def apply_grouped(hidden, bits_combo, order):
    x = hidden.float()
    seq_len = x.shape[1]
    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:G2_END]
    g3_idx = order[G2_END:G3_END]
    g4_idx = order[G3_END:]
    o = x.clone()
    o[..., top_idx] = x[..., top_idx]
    b2, b3, b4 = bits_combo
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        o[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
    nbytes = (TOP_KEEP * seq_len * 16) / 8
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        nbytes += (len(idx) * seq_len * b) / 8 + 4
    return o, nbytes


SCHEMES = [
    ("none", None),
    ("grouped", (4, 4, 4)),
    ("uniform", 4),
    ("grouped", (8, 4, 4)),
    ("grouped", (8, 8, 4)),
    ("uniform", 8),
    ("grouped", (8, 8, 8)),
]

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers


def forward_with_scheme(ids, split, scheme, order):
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    nbytes = None
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None,
                       use_cache=False)
        if split is not None and i == split:
            kind, param = scheme
            od = hidden.dtype
            if kind == "uniform":
                q, nbytes = apply_uniform(hidden, param)
            elif kind == "grouped":
                q, nbytes = apply_grouped(hidden, param, order)
            else:
                q, nbytes = hidden.float(), hidden.shape[1] * hidden.shape[2] * 16 / 8
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0].float(), nbytes


def get_order(ids):
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    act = hs[CALIB_LAYER + 1][0].float()
    return torch.argsort(act.abs().max(dim=0).values, descending=True).to("cuda")


def perplexity(logits, ids):
    lp = F.log_softmax(logits[:-1], dim=-1)
    targets = ids[0, 1:]
    nll = -lp[torch.arange(targets.shape[0]), targets]
    return torch.exp(nll.mean()).item()


def generate(ids, split, scheme, order, n_new=GEN_TOKENS):
    cur = ids.clone()
    for _ in range(n_new):
        with torch.no_grad():
            logits, _ = forward_with_scheme(cur, split, scheme, order)
        cur = torch.cat([cur, logits[-1].argmax().view(1, 1)], dim=1)
    return tok.decode(cur[0, ids.shape[1]:], skip_special_tokens=True)


ppl_ids = tok(PPL_TEXT, return_tensors="pt")["input_ids"][0][:PPL_SEQ_LEN].unsqueeze(0).to("cuda")
order = get_order(ppl_ids)

with torch.no_grad():
    ref_logits, _ = forward_with_scheme(ppl_ids, None, ("none", None), order)
    ref_probs = F.softmax(ref_logits, dim=-1)
    ref_ppl = perplexity(ref_logits, ppl_ids)

out(f"Model: {MODEL_NAME}")
out(f"Layers: {n_layers}  |  perplexity text: {ppl_ids.shape[1]} tokens")
out(f"Group boundaries: top={TOP_KEEP}, shoulder<={G2_END}, mid<={G3_END}")
out(f"Split points: {SPLITS}")
out(f"Reference (uncompressed) perplexity: {ref_ppl:.4f}")
out()

results = {"model": MODEL_NAME, "ref_perplexity": ref_ppl,
           "ppl_seq_len": int(ppl_ids.shape[1]),
           "boundaries": [TOP_KEEP, G2_END, G3_END],
           "splits_tested": SPLITS,
           "gen_prompt": GEN_PROMPT, "splits": {}}

for split in SPLITS:
    out("=" * 78)
    out(f"SPLIT AT LAYER {split}")
    out("=" * 78)
    out(f"{'scheme':18s} {'bytes':>9s} {'%fp16':>7s} {'KL':>10s} "
        f"{'perplexity':>13s} {'PPL rise':>10s}")
    out("-" * 78)
    rec = {}
    for scheme in SCHEMES:
        with torch.no_grad():
            logits, nb = forward_with_scheme(ppl_ids, split, scheme, order)
            probs = F.softmax(logits, dim=-1)
            kl = F.kl_div((probs + 1e-12).log(), ref_probs,
                          reduction="none").sum(dim=-1).mean().item()
            ppl = perplexity(logits, ppl_ids)
        label = f"{scheme[0]}{scheme[1]}" if scheme[0] != "none" else "UNCOMPRESSED"
        fp16_bytes = ppl_ids.shape[1] * model.config.hidden_size * 2
        pct = 100 * nb / fp16_bytes
        rise = 100 * (ppl - ref_ppl) / ref_ppl
        out(f"{label:18s} {nb:9.0f} {pct:6.1f}% {kl:10.4f} "
            f"{ppl:13.4f} {rise:9.2f}%")
        rec[label] = {"bytes": nb, "pct_fp16": pct, "kl": kl,
                      "perplexity": ppl, "ppl_rise_pct": rise}
    results["splits"][str(split)] = rec
    out()

out("=" * 78)
out("GENERATED OUTPUT")
out("=" * 78)
out(f"Prompt: \"{GEN_PROMPT}\"")
out()

gen_ids = tok(GEN_PROMPT, return_tensors="pt")["input_ids"].to("cuda")
gen_order = get_order(gen_ids)
results["generations"] = {}

for split in SPLITS:
    out(f"--- split at layer {split} ---")
    for scheme in SCHEMES:
        label = f"{scheme[0]}{scheme[1]}" if scheme[0] != "none" else "UNCOMPRESSED"
        txt = generate(gen_ids, split, scheme, gen_order).strip()
        r = results["splits"][str(split)][label]
        out()
        out(f"  [{label}]  KL={r['kl']:.4f}  PPL rise={r['ppl_rise_pct']:+.2f}%")
        out(f"  {txt[:300]}")
        results["generations"][f"L{split}_{label}"] = txt
    out()

out("=" * 78)
out("SCHEME SELECTION AT DIFFERENT PERPLEXITY BUDGETS")
out("=" * 78)
out("Cheapest scheme whose perplexity rise stays within each budget.")
out()

PPL_BUDGETS = [1.0, 3.0, 5.0, 10.0, 15.0]
results["budget_picks"] = {}
for split in SPLITS:
    rec = results["splits"][str(split)]
    out(f"Split L{split}:")
    picks = {}
    for B in PPL_BUDGETS:
        ok = [(k, v) for k, v in rec.items()
              if k != "UNCOMPRESSED" and v["ppl_rise_pct"] <= B]
        if ok:
            best = min(ok, key=lambda kv: kv[1]["bytes"])
            out(f"  <= {B:5.1f}% : {best[0]:16s} {best[1]['bytes']:9.0f} B "
                f"({best[1]['pct_fp16']:5.1f}% of fp16)  KL={best[1]['kl']:.4f}")
            picks[str(B)] = {"scheme": best[0], **best[1]}
        else:
            out(f"  <= {B:5.1f}% : no scheme qualifies")
            picks[str(B)] = None
    results["budget_picks"][str(split)] = picks
    out()

os.makedirs(OUT_DIR, exist_ok=True)
json_path = os.path.join(OUT_DIR, f"{RUN_TAG}.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

txt_path = os.path.join(OUT_DIR, f"{RUN_TAG}.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(_lines) + "\n")

print(f"\nSaved: {json_path}")
print(f"Saved: {txt_path}")
