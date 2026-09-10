"""
Does the p95 quality table hold outside its calibration domain?

THE PROBLEM: the percentile table is built entirely from WikiText-2 - English
encyclopaedic prose. If real traffic is harder, the table is optimistic and the
quality budget silently fails. Stated as a limitation but never tested.

METHOD CORRECTION FROM THE FIRST ATTEMPT: that version padded short samples by
REPEATING them to fill 256 tokens. The model had already seen the tokens once,
so perplexity collapsed to ~1.3 against WikiText's 11.5 - the "hard" texts came
out nine times EASIER than the baseline and the experiment tested nothing.
Every passage here stands alone, and any passage under MIN_TOKENS is REJECTED
rather than padded, so the bug cannot recur silently.

DOMAINS:
  shakespeare      archaic English, unusual syntax
  code             punctuation-dense, non-prose token structure
  hindi            non-Latin script, OFFICIALLY SUPPORTED by Llama 3.1
  gujarati         non-Latin script, NOT officially supported
  dense_technical  specialised vocabulary, formal notation
  dialogue         fragmentary, heavy quotation

LANGUAGE NOTE: Meta lists Llama 3.1 as supporting English plus French, German,
Hindi, Italian, Portuguese, Spanish and Thai. Gujarati is not listed, though
Meta notes the model saw a broader set of languages in training. So Hindi tests
a supported non-Latin script and Gujarati tests genuinely out-of-distribution
input. Report Gujarati's result as "an unsupported language", not as a claim
about Gujarati specifically.
"""
import torch, json, os, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
BASELINE = OUT_DIR + "/11a_collect.json"
RUN_TAG = "14_hard_prompts"

SEQ_LEN = 256
MIN_TOKENS = SEQ_LEN + 5
LAYERS = [3, 6, 9, 11]
CALIB_LAYER = 2
TOP_KEEP, G2_END, G3_END = 5, 72, 1312
KL_BUDGETS = [0.05, 0.10, 0.25]

SCHEMES = {"grouped(4,4,4)": (4, 4, 4), "grouped(8,4,4)": (8, 4, 4),
           "grouped(8,8,4)": (8, 8, 4), "grouped(8,8,8)": (8, 8, 8)}
SCHEME_ORDER = list(SCHEMES)

HARD_TEXTS = json.load(open(OUT_DIR + "/14_texts.json"))

_lines = []
def out(s=""):
    print(s, flush=True); _lines.append(s)

def quant_group(x, n):
    if x.numel() == 0 or n >= 16: return x.clone()
    a = x.abs().max()
    if a == 0: return x.clone()
    q = 2 ** (n - 1) - 1
    s = a / q
    return torch.clamp(torch.round(x / s), -q - 1, q) * s

def apply_grouped(hidden, bits, order):
    x = hidden.float()
    o = x.clone()
    o[..., order[:TOP_KEEP]] = x[..., order[:TOP_KEEP]]
    for idx, b in [(order[TOP_KEEP:G2_END], bits[0]),
                   (order[G2_END:G3_END], bits[1]),
                   (order[G3_END:], bits[2])]:
        o[..., idx] = quant_group(x[..., idx].flatten(), b).reshape(x[..., idx].shape)
    return o

print("Loading model...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()

def perplexity(lg, ids):
    lp = F.log_softmax(lg[:-1], dim=-1)
    t = ids[0, 1:]
    return torch.exp((-lp[torch.arange(t.shape[0]), t]).mean()).item()

def pct(v, p):
    v = sorted(v); return v[min(len(v) - 1, int(p * len(v)))]

print("Loading WikiText baseline...", flush=True)
base = json.load(open(BASELINE))
BR = [r for r in base["records"]
      if r["seq_len"] == SEQ_LEN and r["order"] == "per_layer"]
baseline = {}
for L in LAYERS:
    for s in SCHEMES:
        k = [r["kl"] for r in BR if r["layer"] == L and r["scheme"] == s]
        if k:
            baseline[f"{L}_{s}"] = {"mean": sum(k)/len(k), "p95": pct(k, .95),
                                    "max": max(k), "n": len(k)}
bp = [r["ref_ppl"] for r in BR if r["layer"] == LAYERS[0]
      and r["scheme"] == SCHEME_ORDER[0]]
base_ppl = sum(bp) / len(bp)

out(f"Model: {MODEL_NAME}")
out(f"WikiText baseline: {len(bp)} texts, mean reference perplexity {base_ppl:.3f}")
out(f"seq_len {SEQ_LEN}, layers {LAYERS}, per-layer channel ordering")
out(f"Passages shorter than {MIN_TOKENS} tokens are REJECTED, not padded.")
out()

records, rejected = [], []
t0 = time.time()

for domain, samples in HARD_TEXTS.items():
    used = 0
    for si, text in enumerate(samples):
        n_tok = len(tok(text)["input_ids"])
        if n_tok < MIN_TOKENS:
            rejected.append((domain, si, n_tok))
            continue
        used += 1
        ids = tok(text, return_tensors="pt")["input_ids"][0][:SEQ_LEN].unsqueeze(0).to("cuda")
        with torch.no_grad():
            b = model(ids, output_hidden_states=True)
        ref_probs = F.softmax(b.logits[0].float(), dim=-1)
        ref_ppl = perplexity(b.logits[0].float(), ids)
        orders = {L: torch.argsort(b.hidden_states[L+1][0].float().abs().max(dim=0).values,
                                   descending=True).to("cuda") for L in LAYERS}
        del b; torch.cuda.empty_cache()

        for L in LAYERS:
            for sname, bits in SCHEMES.items():
                h = model.model.embed_tokens(ids)
                pos = torch.arange(SEQ_LEN).unsqueeze(0).to("cuda")
                pe = model.model.rotary_emb(h, pos)
                with torch.no_grad():
                    for i, layer in enumerate(model.model.layers):
                        h = layer(h, attention_mask=None, position_ids=pos,
                                  position_embeddings=pe, past_key_values=None,
                                  use_cache=False)
                        if i == L:
                            h = apply_grouped(h, bits, orders[L]).to(h.dtype)
                    h = model.model.norm(h)
                    lg = model.lm_head(h)[0].float()
                p = F.softmax(lg, dim=-1)
                kl = F.kl_div((p + 1e-12).log(), ref_probs,
                              reduction="none").sum(-1).mean().item()
                records.append({"domain": domain, "sample": si, "layer": L,
                                "scheme": sname, "kl": kl, "ref_ppl": ref_ppl,
                                "n_tokens": n_tok})
        del orders; torch.cuda.empty_cache()
    print(f"  {domain}: {used}/{len(samples)} passages used", flush=True)

if rejected:
    out("REJECTED (too short - would have needed padding):")
    for d, i, n in rejected:
        out(f"  {d} sample {i}: {n} tokens < {MIN_TOKENS}")
    out()

DOMAINS = sorted(set(r["domain"] for r in records))
print(f"\n{len(records)} records in {(time.time()-t0)/60:.1f} min\n", flush=True)

out("=" * 92)
out("1. HOW HARD IS EACH DOMAIN?")
out("=" * 92)
out("Reference perplexity of the UNCOMPRESSED model. Higher = harder to")
out("predict. If these are not above the WikiText figure, the test has failed")
out("to produce hard input and nothing below is meaningful.")
out()
out(f"{'domain':>18} {'mean ref_ppl':>14} {'vs WikiText':>13}")
out("-" * 92)
out(f"{'wikitext (calib)':>18} {base_ppl:14.3f} {'1.00x':>13}")
dppl = {}
for d in DOMAINS:
    v = [r["ref_ppl"] for r in records if r["domain"] == d
         and r["layer"] == LAYERS[0] and r["scheme"] == SCHEME_ORDER[0]]
    dppl[d] = sum(v) / len(v)
    out(f"{d:>18} {dppl[d]:14.3f} {dppl[d]/base_ppl:12.2f}x")
out()

out("=" * 92)
out("2. WOULD THE CONTROLLER'S CHOICE BREACH THE BUDGET?")
out("=" * 92)
out("The controller picks the cheapest scheme whose WikiText p95 meets the")
out("budget. This asks whether that choice ACTUALLY meets the budget on each")
out("domain. Not whether the KL differs - whether the DECISION is wrong.")
out()
wrong = total = 0
for budget in KL_BUDGETS:
    out(f"--- budget KL <= {budget} ---")
    out(f"{'layer':>5} {'controller picks':>17} {'wiki p95':>10}  " +
        "".join(f"{d[:9]:>11}" for d in DOMAINS))
    out("-" * 92)
    for L in LAYERS:
        pick = next((s for s in SCHEME_ORDER
                     if f"{L}_{s}" in baseline and baseline[f"{L}_{s}"]["p95"] <= budget),
                    None)
        if not pick:
            out(f"{L:5d} {'none passes':>17}"); continue
        line = f"{L:5d} {pick:>17} {baseline[f'{L}_{pick}']['p95']:10.5f}  "
        for d in DOMAINS:
            v = [r["kl"] for r in records if r["domain"] == d
                 and r["layer"] == L and r["scheme"] == pick]
            m = sum(v) / len(v)
            total += 1
            if m > budget:
                wrong += 1; line += f"{m:10.5f}X"
            else:
                line += f"{m:10.5f} "
        out(line)
    out()
out("  X = the controller's choice BREACHES the budget on that domain")
out()

out("=" * 92)
out("VERDICT")
out("=" * 92)
harder = [d for d in DOMAINS if dppl[d] > base_ppl]
out(f"  Domains harder than WikiText: {len(harder)} of {len(DOMAINS)}"
    + (f"  ({', '.join(harder)})" if harder else ""))
if harder:
    h = max(harder, key=lambda d: dppl[d])
    out(f"  Hardest: {h} at {dppl[h]:.2f} ({dppl[h]/base_ppl:.2f}x WikiText)")
out(f"  Controller decisions that breached the budget: {wrong} of {total}")
out()
if not harder:
    out("  TEST INCONCLUSIVE. No domain came out harder than the calibration")
    out("  corpus, so a clean pass says little. Do not claim the table")
    out("  generalises to hard input on this evidence.")
elif wrong == 0:
    out("  THE TABLE HOLDS. Every scheme the controller would select meets the")
    out("  budget on every domain tested, including ones harder than the")
    out("  calibration corpus. The p95 margin absorbs the difference.")
    out("  The quality guarantee is not confined to the calibration domain.")
elif wrong / total < 0.15:
    out("  MOSTLY HOLDS, with isolated breaches. The design is usable but the")
    out("  guarantee is domain-dependent - state that rather than claiming")
    out("  general coverage.")
else:
    out("  DOES NOT GENERALISE. A substantial fraction of the controller's")
    out("  choices breach the budget on other domains. The p95 is valid only")
    out("  within the calibration domain; deployment would need")
    out("  domain-specific calibration or a more conservative percentile.")

os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"model": MODEL_NAME, "seq_len": SEQ_LEN, "layers": LAYERS,
           "wikitext_baseline": baseline, "wikitext_mean_ref_ppl": base_ppl,
           "domain_ref_ppl": dppl, "rejected": rejected,
           "wrong_decisions": wrong, "total_decisions": total,
           "records": records},
          open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w"), indent=2)
open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w").write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.{{json,txt}}")
