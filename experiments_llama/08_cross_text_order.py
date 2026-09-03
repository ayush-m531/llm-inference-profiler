"""
Cross-text validation of the per-layer channel order table.

07 and 07_01 each derived the channel orders from the SAME text they then
tested on. That is an oracle - a deployed system computes orders offline from
calibration data and applies them to unseen input.

This experiment removes the oracle:
  - orders are built from CALIBRATION text
  - they are then applied to two different EVALUATION texts
  - compared against (a) the single frozen order and (b) the oracle order
    derived from the evaluation text itself

If cross-text orders perform close to oracle orders, the precomputed-table
design is validated. If they collapse toward the frozen order, the design does
not survive deployment.

Writes JSON and a TXT transcript.
"""
import torch
import json
import os
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR = "/home/ayush.thakar/thesis/experiments_llama/results"
RUN_TAG = "08_cross_text_order"

SPLITS = list(range(0, 12))
TOP_KEEP = 5
G2_END = 72
G3_END = 1312
CALIB_LAYER = 2
PPL_SEQ_LEN = 256
PPL_BUDGETS = [1.0, 3.0, 5.0, 10.0]

# Text the orders are DERIVED from. Deliberately different in character from
# both evaluation texts, to make this a fair test.
CALIB_TEXT = ("A recipe for a simple loaf begins with flour, water, salt and yeast. "
              "The baker mixes them until a rough dough forms, then rests it so the "
              "flour can absorb the water fully. Kneading develops the gluten network "
              "that traps gas during fermentation. After the first rise the dough is "
              "shaped and left to prove a second time, more briefly. A hot oven with "
              "steam in the early minutes lets the loaf expand before the crust sets. "
              "Bakers judge doneness by colour and by tapping the base, listening for "
              "a hollow sound. Cooling on a rack prevents the trapped steam from "
              "softening the crust. Variations in hydration, flour type and "
              "fermentation time produce very different results from the same four "
              "ingredients, which is why bread rewards practice more than precision. ") * 4

# Evaluation texts - the same two used in 07 and 07_01
EVAL_TEXTS = {
    "technical": ("Climate scientists have documented a steady rise in global average "
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
                  "should shape the distribution of costs and obligations. ") * 3,

    "narrative": ("The construction of the railway across the mountains took eleven years "
                  "and cost more lives than anyone had anticipated. Workers arrived from "
                  "distant provinces, drawn by wages that seemed generous until they "
                  "discovered the price of food at the company stores. Winters halted "
                  "progress entirely; men huddled in timber barracks while snow buried the "
                  "half finished tunnels. In spring the work resumed, and with it the "
                  "accidents: rockfalls, blasting misjudgements, the slow poisoning of men "
                  "who breathed dust for twelve hours a day. The engineers kept meticulous "
                  "records of gradients and expenditure, and almost none of the dead. When "
                  "the final section opened, dignitaries travelled the route in decorated "
                  "carriages and made speeches about progress and national unity. The "
                  "villages along the line prospered for a generation, then declined again "
                  "as roads improved and freight moved elsewhere. ") * 3,
}


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
    ("grouped", (4, 4, 4)),
    ("grouped", (8, 4, 4)),
    ("grouped", (8, 8, 4)),
    ("grouped", (8, 8, 8)),
]

print("Loading model...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
model.eval()
n_layers = model.config.num_hidden_layers
H = model.config.hidden_size


def build_orders(text):
    """Per-layer channel orders derived from one text."""
    ids = tok(text, return_tensors="pt")["input_ids"][0][:PPL_SEQ_LEN].unsqueeze(0).to("cuda")
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    ords = {}
    for L in range(n_layers):
        act = hs[L + 1][0].float()
        ords[L] = torch.argsort(act.abs().max(dim=0).values,
                                descending=True).to("cuda")
    del hs
    return ords


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
            od = hidden.dtype
            q, nbytes = apply_grouped(hidden, scheme[1], order)
            hidden = q.to(od)
    hidden = model.model.norm(hidden)
    return model.lm_head(hidden)[0].float(), nbytes


def perplexity(logits, ids):
    lp = F.log_softmax(logits[:-1], dim=-1)
    targets = ids[0, 1:]
    nll = -lp[torch.arange(targets.shape[0]), targets]
    return torch.exp(nll.mean()).item()


def group_of(order):
    g = torch.empty(H, dtype=torch.long, device=order.device)
    g[order[:TOP_KEEP]] = 0
    g[order[TOP_KEEP:G2_END]] = 1
    g[order[G2_END:G3_END]] = 2
    g[order[G3_END:]] = 3
    return g


print("Building orders from CALIBRATION text...")
calib_orders = build_orders(CALIB_TEXT)

out(f"Model: {MODEL_NAME}")
out(f"Orders derived from: calibration text (recipe/instructional prose)")
out(f"Evaluated on: {list(EVAL_TEXTS.keys())}")
out(f"Split range: L{SPLITS[0]}-L{SPLITS[-1]}")
out(f"Noise floor: treat PPL differences under ~1pp as indistinguishable.")
out()

results = {"model": MODEL_NAME, "calib_layer": CALIB_LAYER,
           "splits": SPLITS, "eval_texts": {}}

for tname, ttext in EVAL_TEXTS.items():
    out("=" * 78)
    out(f"EVALUATION TEXT: {tname}")
    out("=" * 78)

    ids = tok(ttext, return_tensors="pt")["input_ids"][0][:PPL_SEQ_LEN].unsqueeze(0).to("cuda")
    oracle_orders = build_orders(ttext)      # the unrealistic best case
    frozen = calib_orders[CALIB_LAYER]        # single order, from calibration

    with torch.no_grad():
        ref_logits, _ = forward_with_scheme(ids, None, None, frozen)
        ref_probs = F.softmax(ref_logits, dim=-1)
        ref_ppl = perplexity(ref_logits, ids)
    out(f"Reference (uncompressed) perplexity: {ref_ppl:.4f}")
    out()

    # how far do calibration orders sit from oracle orders?
    out("ORDER AGREEMENT: calibration-derived vs oracle (same-text) orders")
    out(f"{'layer':>5} {'same group':>12} {'top5 kept':>11} {'top72 kept':>12}")
    out("-" * 45)
    agree = {}
    for L in SPLITS:
        gc = group_of(calib_orders[L])
        go = group_of(oracle_orders[L])
        same = (gc == go).float().mean().item()
        k5 = len(set(calib_orders[L][:TOP_KEEP].tolist()) &
                 set(oracle_orders[L][:TOP_KEEP].tolist())) / TOP_KEEP
        k72 = len(set(calib_orders[L][:G2_END].tolist()) &
                  set(oracle_orders[L][:G2_END].tolist())) / G2_END
        out(f"{L:5d} {100*same:11.2f}% {100*k5:10.1f}% {100*k72:11.1f}%")
        agree[L] = {"same_group": same, "top5": k5, "top72": k72}
    out()

    out("QUALITY: frozen vs calibration-derived per-layer vs oracle per-layer")
    out()
    rec_all = {}
    for L in SPLITS:
        out(f"--- split L{L} ---")
        out(f"{'scheme':18s} {'bytes':>9s} | {'PPL froz':>9s} {'PPL calib':>10s} "
            f"{'PPL oracle':>11s} | {'calib gain':>11s} {'vs oracle':>10s}")
        out("-" * 82)
        rec = {}
        for scheme in SCHEMES:
            with torch.no_grad():
                lg_f, nb = forward_with_scheme(ids, L, scheme, frozen)
                ppl_f = perplexity(lg_f, ids)
                lg_c, _ = forward_with_scheme(ids, L, scheme, calib_orders[L])
                ppl_c = perplexity(lg_c, ids)
                lg_o, _ = forward_with_scheme(ids, L, scheme, oracle_orders[L])
                ppl_o = perplexity(lg_o, ids)
            rf = 100 * (ppl_f - ref_ppl) / ref_ppl
            rc = 100 * (ppl_c - ref_ppl) / ref_ppl
            ro = 100 * (ppl_o - ref_ppl) / ref_ppl
            label = f"grouped{scheme[1]}"
            out(f"{label:18s} {nb:9.0f} | {rf:8.2f}% {rc:9.2f}% {ro:10.2f}% "
                f"| {rf-rc:10.2f}pp {rc-ro:9.2f}pp")
            rec[label] = {"bytes": nb, "ppl_frozen": rf,
                          "ppl_calib": rc, "ppl_oracle": ro}
        rec_all[str(L)] = rec
        out()

    # scheme selection under each ordering strategy
    out("SCHEME SELECTED AT EACH BUDGET")
    out()
    changed_vs_frozen = 0
    matched_oracle = 0
    total = 0
    savings = []
    for B in PPL_BUDGETS:
        out(f"--- budget <= {B}% ---")
        out(f"{'layer':>5} {'frozen':>18} {'calib':>18} {'oracle':>18} "
            f"{'saved vs froz':>14}")
        out("-" * 78)
        for L in SPLITS:
            rec = rec_all[str(L)]

            def pick(key):
                ok = [(k, v) for k, v in rec.items() if v[key] <= B]
                return min(ok, key=lambda kv: kv[1]["bytes"]) if ok else None

            pf, pc, po = pick("ppl_frozen"), pick("ppl_calib"), pick("ppl_oracle")
            lf = pf[0] if pf else "none"
            lc = pc[0] if pc else "none"
            lo = po[0] if po else "none"
            saved = (pf[1]["bytes"] - pc[1]["bytes"]) if (pf and pc) else 0
            total += 1
            if lf != lc:
                changed_vs_frozen += 1
                if saved > 0:
                    savings.append(100 * saved / pf[1]["bytes"])
            if lc == lo:
                matched_oracle += 1
            mark = "  <--" if lf != lc else ""
            out(f"{L:5d} {lf:>18} {lc:>18} {lo:>18} {saved:14.0f}{mark}")
        out()

    out(f"SUMMARY for {tname}:")
    out(f"  calibration orders changed the scheme vs frozen in "
        f"{changed_vs_frozen}/{total} cases")
    if savings:
        out(f"  savings where changed: {min(savings):.1f}% to {max(savings):.1f}% "
            f"(mean {sum(savings)/len(savings):.1f}%)")
    out(f"  calibration orders matched the ORACLE choice in "
        f"{matched_oracle}/{total} cases "
        f"({100*matched_oracle/total:.0f}%)")
    out()

    results["eval_texts"][tname] = {
        "ref_perplexity": ref_ppl, "order_agreement": agree,
        "quality": rec_all, "changed_vs_frozen": changed_vs_frozen,
        "matched_oracle": matched_oracle, "total_cases": total,
        "savings_pct": savings}

out("=" * 78)
out("VERDICT")
out("=" * 78)
out("If calibration-derived orders match the oracle in most cases and still")
out("beat the frozen order, the precomputed-table design survives deployment:")
out("orders can be built offline and shipped, with no per-request cost.")
out("If they collapse toward the frozen order, the design does not hold.")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.json"), "w") as f:
    json.dump(results, f, indent=2)
with open(os.path.join(OUT_DIR, f"{RUN_TAG}.txt"), "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nSaved: {OUT_DIR}/{RUN_TAG}.json")
print(f"Saved: {OUT_DIR}/{RUN_TAG}.txt")
