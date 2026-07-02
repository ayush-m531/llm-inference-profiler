import torch
import json
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Hello, my name is"
N_BITS = 4


def quantize_dequantize(x, n_bits):
    absmax = x.abs().max()
    if absmax == 0:
        return x.clone()
    qmax = 2 ** (n_bits - 1) - 1
    scale = absmax / qmax
    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax - 1, qmax)
    return q * scale


def run_from_embeddings(model, input_ids, quant_layer=None, n_bits=N_BITS):
    # Runs the full model manually from embeddings to logits.
    # If quant_layer is set, that layer's OUTPUT activation is quantized
    # before being passed on (simulating quantization at that split point).
    # Returns final logits for the last token position.
    hidden = model.model.embed_tokens(input_ids)
    pos = torch.arange(input_ids.shape[1]).unsqueeze(0).to(input_ids.device)
    pos_emb = model.model.rotary_emb(hidden, pos)

    for i, layer in enumerate(model.model.layers):
        hidden = layer(
            hidden,
            attention_mask=None,
            position_ids=pos,
            position_embeddings=pos_emb,
            past_key_values=None,
            use_cache=False,
        )
        # quantize this layer's output if it's the target layer
        if quant_layer is not None and i == quant_layer:
            orig_dtype = hidden.dtype
            hidden = quantize_dequantize(hidden.float(), n_bits).to(orig_dtype)

    # final norm + language model head -> logits
    hidden = model.model.norm(hidden)
    logits = model.lm_head(hidden)
    # logits shape: [1, seq_len, vocab]; take last token's logits
    return logits[0, -1, :].float()


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
model.eval()
print(f"Model loaded. Layers: {model.config.num_hidden_layers}")

inputs = tokenizer(PROMPT, return_tensors="pt")
input_ids = inputs["input_ids"].to("cuda")

with torch.no_grad():
    # 1. Reference logits (no quantization anywhere)
    ref_logits = run_from_embeddings(model, input_ids, quant_layer=None)
    ref_probs = F.softmax(ref_logits, dim=-1)
    ref_top1 = torch.argmax(ref_logits).item()
    ref_top1_token = tokenizer.decode([ref_top1])
    print(f"Reference top-1 next token: '{ref_top1_token}' (id {ref_top1})\n")

    results = []
    # 2. Quantize each layer one at a time, measure damage to logits
    for layer_idx in range(model.config.num_hidden_layers):
        dmg_logits = run_from_embeddings(model, input_ids, quant_layer=layer_idx)
        dmg_probs = F.softmax(dmg_logits, dim=-1)

        # KL divergence: how different is the whole probability distribution
        # KL(ref || damaged) — add small epsilon for numerical safety
        kl = F.kl_div(
            (dmg_probs + 1e-12).log(),
            ref_probs,
            reduction="sum",
        ).item()

        # top-1 agreement: does the model still pick the same next token?
        dmg_top1 = torch.argmax(dmg_logits).item()
        top1_same = (dmg_top1 == ref_top1)

        results.append({
            "layer": layer_idx,
            "kl_divergence": kl,
            "top1_same": top1_same,
            "dmg_top1_token": tokenizer.decode([dmg_top1]),
        })

        flag = "same" if top1_same else f"CHANGED -> '{tokenizer.decode([dmg_top1])}'"
        print(f"Quantize layer {layer_idx:02d} | KL: {kl:.6f} | top-1: {flag}")

os.makedirs("/home/ayush.thakar/thesis/experiments/quantization/results", exist_ok=True)
results_path = "/home/ayush.thakar/thesis/experiments/quantization/results/06_endtoend_validation.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")

layers = [r["layer"] for r in results]
kls = [r["kl_divergence"] for r in results]
colors = ["seagreen" if r["top1_same"] else "indianred" for r in results]

plt.figure(figsize=(13, 6))
plt.bar(layers, kls, color=colors)
plt.xlabel("Quantized Layer Index")
plt.ylabel("KL Divergence (ref || quantized)")
plt.title(f"End-to-End Output Damage from Quantizing Each Layer ({N_BITS}-bit)\n"
          f"(red = top-1 next token changed)")
plt.xticks(layers)
plt.tight_layout()
plot_path = "/home/ayush.thakar/thesis/experiments/quantization/results/06_endtoend_validation.png"
plt.savefig(plot_path, dpi=150)
print(f"Plot saved to {plot_path}")
