"""
Bulk measurement collection: activation statistics and quality cost across
many texts, layers, schemes and sequence lengths.

PURPOSE - three things this data supports:
  1. A PERCENTILE-BASED LOOKUP TABLE. The controller's current table is
     hardcoded from ONE text. With hundreds of samples per cell it can select
     on the 95th percentile instead, meeting the quality budget for 95% of
     inputs rather than for the single input that happened to be measured.
  2. THE HARDNESS-PREDICTOR QUESTION. Can a cheap activation statistic predict
     per-input quality cost? Requires correlation measured WITHIN a fixed
     (layer, scheme, seq_len) - pooling across layers would merely recover the
     already-known layer effect.
  3. SEQUENCE-LENGTH DEPENDENCE. All prior quality work used a single length.
     Whether the lookup table needs a length dimension is untested.

This script ONLY COLLECTS. Analysis is separate (11b) so the data can be
re-analysed without repeating the GPU work.

Both the frozen and per-layer channel orderings are recorded, so the ordering
result gets a large sample too (previously measured on two texts).
"""
import torch
import json
import os
import time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "11a_collect"

N_TEXTS = 300
SEQ_LENS = [128, 256, 512]
LAYERS = list(range(0, 12))          # the memory-feasible split range
CALIB_LAYER = 2

TOP_KEEP, G2_END, G3_END = 5, 72, 1312

SCHEMES = {
    "grouped(4,4,4)": (4, 4, 4),
    "grouped(8,4,4)": (8, 4, 4),
    "grouped(8,8,4)": (8, 8, 4),
    "grouped(8,8,8)": (8, 8, 8),
}

SAVE_EVERY = 25                      # texts between partial saves


def quant_group(x, n_bits):
    if x.numel() == 0 or n_bits >= 16:
        return x.clone()
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    return torch.clamp(torch.round(x / scale), -qmax - 1, qmax) * scale


def apply_grouped(hidden, bits, order):
    x = hidden.float()
    top_idx = order[:TOP_KEEP]
    g2 = order[TOP_KEEP:G2_END]
    g3 = order[G2_END:G3_END]
    g4 = order[G3_END:]
    o = x.clone()
    o[..., top_idx] = x[..., top_idx]
    b2, b3, b4 = bits
    for idx, b in [(g2, b2), (g3, b3), (g4, b4)]:
        o[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
    return o


def scheme_bytes(seq_len, bits, hidden_size):
    b2, b3, b4 = bits
    nb = TOP_KEEP * seq_len * 16 / 8
    for n, b in [(G2_END - TOP_KEEP, b2), (G3_END - G2_END, b3),
                 (hidden_size - G3_END, b4)]:
        nb += n * seq_len * b / 8 + 4
    return nb


print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
H = model.config.hidden_size
MAX_SEQ = max(SEQ_LENS)


def load_texts(n):
    """WikiText: the standard perplexity benchmark. wikitext-2 test is small,
    so fall back to wikitext-103 if it cannot supply enough long passages."""
    from datasets import load_dataset
    need = MAX_SEQ + 20

    for repo_id, cfg, split in [
            ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
            ("Salesforce/wikitext", "wikitext-103-raw-v1", "test"),
            ("Salesforce/wikitext", "wikitext-2-raw-v1", "validation")]:
        try:
            print(f"Trying {repo_id} / {cfg} / {split} ...", flush=True)
            ds = load_dataset(repo_id, cfg, split=split)
            buf, texts = "", []
            for row in ds:
                t = row["text"].strip()
                if not t or t.startswith("="):
                    continue
                buf += " " + t
                if len(tok(buf)["input_ids"]) >= need:
                    texts.append(buf.strip())
                    buf = ""
                    if len(texts) >= n:
                        break
            print(f"  got {len(texts)} passages", flush=True)
            if len(texts) >= n:
                return texts, f"{repo_id}/{cfg}/{split}"
            if len(texts) >= 50:
                print(f"  fewer than requested but usable; continuing",
                      flush=True)
                return texts, f"{repo_id}/{cfg}/{split}"
        except Exception as e:
            print(f"  failed: {str(e)[:120]}", flush=True)

    raise SystemExit("No usable text source found.")


texts, SOURCE = load_texts(N_TEXTS)
print(f"Using {len(texts)} passages from {SOURCE}\n", flush=True)


def perplexity(logits, ids):
    lp = F.log_softmax(logits[:-1], dim=-1)
    tgt = ids[0, 1:]
    return torch.exp((-lp[torch.arange(tgt.shape[0]), tgt]).mean()).item()


def stats_of(act):
    """Statistics the edge could cheaply compute on an activation it already
    holds. act is [seq, hidden] float32 on GPU."""
    v = act.reshape(-1).double()
    mean = v.mean()
    c = v - mean
    var = (c ** 2).mean()
    kurt = ((c ** 4).mean() / (var ** 2)).item()
    skew = ((c ** 3).mean() / (var ** 1.5)).item()

    a = act.abs().double()
    per_ch = a.max(dim=0).values
    tot = per_ch.sum().item()
    n_ch = per_ch.numel()
    srt = torch.sort(per_ch, descending=True).values
    k1 = max(1, int(round(n_ch * 0.01)))
    k5 = max(1, int(round(n_ch * 0.05)))

    return {"kurtosis": kurt,
            "skew": skew,
            "top1pct": srt[:k1].sum().item() / tot,
            "top5pct": srt[:k5].sum().item() / tot,
            "top5_channels_share": srt[:5].sum().item() / tot,
            "max_abs": a.max().item(),
            "mean_abs": a.mean().item(),
            "max_over_mean": (a.max() / a.mean()).item(),
            "std": var.sqrt().item()}


records = []
t_start = time.time()
os.makedirs(OUT_DIR, exist_ok=True)

expected = len(texts) * len(SEQ_LENS) * len(LAYERS) * len(SCHEMES) * 2
print(f"Target: {expected:,} measurements "
      f"({len(texts)} texts x {len(SEQ_LENS)} lengths x {len(LAYERS)} layers "
      f"x {len(SCHEMES)} schemes x 2 orderings)\n", flush=True)


def save(partial=True):
    payload = {"model": MODEL_NAME, "n_texts": len(texts),
               "seq_lens": SEQ_LENS, "layers": LAYERS,
               "schemes": list(SCHEMES), "calib_layer": CALIB_LAYER,
               "boundaries": [TOP_KEEP, G2_END, G3_END],
               "source": SOURCE, "complete": not partial,
               "records": records}
    name = f"{RUN_TAG}_partial.json" if partial else f"{RUN_TAG}.json"
    with open(os.path.join(OUT_DIR, name), "w") as f:
        json.dump(payload, f)


for ti, text in enumerate(texts):
    all_ids = tok(text, return_tensors="pt")["input_ids"][0]

    for seq_len in SEQ_LENS:
        if all_ids.shape[0] < seq_len:
            continue
        ids = all_ids[:seq_len].unsqueeze(0).to("cuda")

        with torch.no_grad():
            base = model(ids, output_hidden_states=True)
        ref_logits = base.logits[0].float()
        ref_probs = F.softmax(ref_logits, dim=-1)
        ref_ppl = perplexity(ref_logits, ids)

        orders = {}
        for L in LAYERS:
            act = base.hidden_states[L + 1][0].float()
            orders[L] = torch.argsort(act.abs().max(dim=0).values,
                                      descending=True).to("cuda")
        frozen = orders[CALIB_LAYER]
        layer_stats = {L: stats_of(base.hidden_states[L + 1][0].float())
                       for L in LAYERS}
        del base
        torch.cuda.empty_cache()

        for L in LAYERS:
            for sname, bits in SCHEMES.items():
                for otype, order in (("frozen", frozen),
                                     ("per_layer", orders[L])):
                    hidden = model.model.embed_tokens(ids)
                    pos = torch.arange(seq_len).unsqueeze(0).to("cuda")
                    pe = model.model.rotary_emb(hidden, pos)
                    with torch.no_grad():
                        for i, layer in enumerate(model.model.layers):
                            hidden = layer(hidden, attention_mask=None,
                                           position_ids=pos,
                                           position_embeddings=pe,
                                           past_key_values=None,
                                           use_cache=False)
                            if i == L:
                                hidden = apply_grouped(
                                    hidden, bits, order).to(hidden.dtype)
                        hidden = model.model.norm(hidden)
                        lg = model.lm_head(hidden)[0].float()

                    p = F.softmax(lg, dim=-1)
                    kl = F.kl_div((p + 1e-12).log(), ref_probs,
                                  reduction="none").sum(-1).mean().item()
                    ppl = perplexity(lg, ids)

                    rec = {"text_id": ti, "seq_len": seq_len, "layer": L,
                           "scheme": sname, "order": otype,
                           "bytes": scheme_bytes(seq_len, bits, H),
                           "ref_ppl": ref_ppl, "kl": kl, "ppl": ppl,
                           "ppl_rise_pct": 100 * (ppl - ref_ppl) / ref_ppl}
                    rec.update(layer_stats[L])
                    records.append(rec)

        del orders, frozen
        torch.cuda.empty_cache()

    if (ti + 1) % SAVE_EVERY == 0:
        el = time.time() - t_start
        rate = (ti + 1) / el
        eta = (len(texts) - ti - 1) / rate / 60 if rate > 0 else 0
        print(f"  {ti+1}/{len(texts)} texts | {len(records):,} records | "
              f"{el/60:.1f} min elapsed | ~{eta:.0f} min left", flush=True)
        save(partial=True)

save(partial=False)
el = time.time() - t_start
print(f"\nDone. {len(records):,} records in {el/60:.1f} minutes.", flush=True)
print(f"Source: {SOURCE}", flush=True)
print(f"Saved: {OUT_DIR}/{RUN_TAG}.json", flush=True)

print("\nSanity check - KL by layer "
      "(per_layer order, seq 256, grouped(8,4,4)):", flush=True)
for L in LAYERS:
    rows = [r for r in records if r["layer"] == L and r["seq_len"] == 256
            and r["scheme"] == "grouped(8,4,4)" and r["order"] == "per_layer"]
    if rows:
        ks = sorted(r["kl"] for r in rows)
        print(f"  L{L:2d}: mean {sum(ks)/len(ks):.5f}  min {ks[0]:.5f}  "
              f"max {ks[-1]:.5f}  p95 {ks[int(0.95*len(ks))]:.5f}  "
              f"n={len(ks)}", flush=True)
