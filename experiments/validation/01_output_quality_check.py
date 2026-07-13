import torch
import json
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"
TEST_SPLITS = [1, 3, 10, 20, 22]     # calm, storm, storm, storm, calm
GEN_TOKENS = 20

TOP_KEEP = 10
G2_END = 40
G3_END = 150

RESULTS_DIR = "/home/ayush.thakar/thesis/experiments/validation/results"


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
        return x.clone()
    return quant_group(x.flatten(), n_bits).reshape(x.shape)


def apply_grouped(hidden, bits_combo):
    x = hidden.float()
    per_channel_max = x.abs().max(dim=1).values.flatten()
    order = torch.argsort(per_channel_max, descending=True)
    top_idx = order[:TOP_KEEP]
    g2_idx = order[TOP_KEEP:G2_END]
    g3_idx = order[G2_END:G3_END]
    g4_idx = order[G3_END:]
    out = x.clone()
    out[..., top_idx] = x[..., top_idx]
    b2, b3, b4 = bits_combo
    for idx, b in [(g2_idx, b2), (g3_idx, b3), (g4_idx, b4)]:
        out[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
    return out


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
print(f"Model loaded. Layers: {n_layers}\n")


def forward_with_scheme(ids, quant_split, scheme):
    hidden = model.model.embed_tokens(ids)
    pos = torch.arange(ids.shape[1]).unsqueeze(0).to("cuda")
    pos_emb = model.model.rotary_emb(hidden, pos)
    for i, layer in enumerate(model.model.layers):
        hidden = layer(hidden, attention_mask=None, position_ids=pos,
                       position_embeddings=pos_emb, past_key_values=None, use_cache=False)
        if quant_split is not None and i == quant_split:
            od = hidden.dtype
            kind, param = scheme
            if kind == "uniform":
                q = apply_uniform(hidden, param)
            elif kind == "grouped":
                q = apply_grouped(hidden, param)
            else:
                q = hidden.float()
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0, -1, :].float()


def generate(prompt_ids, quant_split, scheme, n_new):
    ids = prompt_ids.clone()
    for _ in range(n_new):
        logits = forward_with_scheme(ids, quant_split, scheme)
        next_id = torch.argmax(logits).item()
        ids = torch.cat([ids, torch.tensor([[next_id]]).to("cuda")], dim=1)
    return ids


def top5(logits):
    probs = F.softmax(logits, dim=-1)
    vals, idxs = torch.topk(probs, 5)
    return [(tokenizer.decode([i.item()]), v.item()) for v, i in zip(vals, idxs)]


SCHEMES = [
    ("none",    None),
    ("grouped", (8, 8, 8)),
    ("grouped", (8, 8, 4)),
    ("grouped", (8, 4, 4)),
    ("grouped", (4, 4, 4)),
    ("uniform", 8),
    ("uniform", 4),
]

ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to("cuda")
print(f"PROMPT: '{PROMPT}'\n")

with torch.no_grad():
    ref_logits = forward_with_scheme(ids, None, ("none", None))
    ref_probs = F.softmax(ref_logits, dim=-1)
    ref_gen = generate(ids, None, ("none", None), GEN_TOKENS)
    ref_text = tokenizer.decode(ref_gen[0], skip_special_tokens=True)

    print("=" * 78)
    print("REFERENCE (no quantization)")
    print("=" * 78)
    print("  Top-5 next tokens:")
    for tok, p in top5(ref_logits):
        print(f"    '{tok}' : {p:.4f}")
    print(f"\n  Generated: {ref_text}")
    print()

    results = {"prompt": PROMPT, "reference_text": ref_text, "splits": {}}

    for split in TEST_SPLITS:
        zone = "calm" if (split <= 1 or split >= 21) else "STORM"
        print("\n" + "=" * 78)
        print(f"SPLIT AT LAYER {split}  ({zone} zone)")
        print("=" * 78)
        split_rec = {}

        for scheme in SCHEMES:
            if scheme[0] == "none":
                continue
            logits = forward_with_scheme(ids, split, scheme)
            probs = F.softmax(logits, dim=-1)
            kl = F.kl_div((probs + 1e-12).log(), ref_probs, reduction="sum").item()

            gen = generate(ids, split, scheme, GEN_TOKENS)
            text = tokenizer.decode(gen[0], skip_special_tokens=True)

            label = f"{scheme[0]}{scheme[1]}"
            print(f"\n  --- {label}  (KL = {kl:.4f}) ---")
            print("    Top-5:", ", ".join(f"'{t}':{p:.3f}" for t, p in top5(logits)))
            print(f"    Generated: {text}")

            split_rec[label] = {"kl": kl, "text": text,
                                "top5": [[t, p] for t, p in top5(logits)]}
        results["splits"][str(split)] = split_rec

os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f"{RESULTS_DIR}/01_output_quality_check.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n\nResults saved to {RESULTS_DIR}/01_output_quality_check.json")
