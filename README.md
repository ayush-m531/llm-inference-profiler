# LLM Inference Profiler

Experiments behind an M.Tech thesis on compute–communication co-optimization
for edge–cloud LLM inference.

The question: when a large language model is split between a memory-constrained
device and a cloud server, the intermediate activation has to cross the
network. How do you compress it without destroying the output, and does it
matter where you split?

Two models. **Llama-3.1-8B-Instruct** was always the target;
**Qwen2.5-0.5B-Instruct** is where the work started, because at that point
there was no GPU available and a 0.5B model runs where an 8B one does not. Once
an A100 became available the work moved to Llama. All Llama measurements are on
an NVIDIA A100-40GB.

That sequence turned out to matter. Qwen's activation structure is not merely a
smaller version of Llama's — it is a different shape, and the difference makes
one of the two research questions unanswerable on Qwen. See below.

The system built on these results lives in a companion repository.

---

## Start here

If you are reading this to evaluate the work, three results carry it:

**Outlier-aware compression works and uniform quantization does not.**
Per-tensor uniform quantization produces degenerate output at any split point
tested — on Llama, perplexity rises of 10⁵ to 10⁷ and text like
`">>> ; ; ; ;"`; on Qwen, Chinese-character garbage where grouped 4-bit at
*half* the byte cost produced perfect English. At equal size the difference is
13–99× in KL divergence. This is not a compression-ratio improvement; it is a
feasibility difference.

**Per-layer channel orderings, computed offline, cut quantization error by up
to 749×.** Using one frozen ordering everywhere is catastrophic at layer 0,
where the outlier channels are a different set entirely. Orderings derived from
unrelated calibration text reproduce a same-text oracle's scheme selection in
88–94% of cases, so they can be shipped with the model rather than transmitted.

**Per-input quality cost cannot be predicted from the activation.** It varies
about 8× at a fixed layer and scheme, but five independent approaches across
300 texts found nothing usable. That negative result is what justifies
provisioning for the variation rather than detecting it.

---

## Repository layout

```
experiments/
  quantization/    Qwen: activation structure, group boundaries, how to spend
                   bits, KL budget sweeps
  split_network/   Qwen: split-point selection, early controller
  validation/      Qwen: end-to-end generated-text checks
experiments_llama/ Llama 3.1 8B: everything from the scale-up onward
RESULTS.md         every measured number, with the experiment that produced it
```

Every experiment writes both a `.json` (machine-readable) and a `.txt`
(readable transcript) into its `results/` directory.

---

## The Qwen phase

Qwen2.5-0.5B-Instruct (24 layers, hidden 896) is where the work began, before
GPU access. Everything about the compression method was worked out here and
those results still stand. What did *not* transfer was the split-point question
— for a structural reason worth understanding.

`experiments/quantization/`, `experiments/split_network/`,
`experiments/validation/`

### Activation structure has three zones

| Layers | Kurtosis | Top 1% of channels hold |
|---|---|---|
| 0–1 | 31–57 | ~5% |
| 2–20 | 68,000–84,000 | 47–50% |
| 21–23 | 88–258 | 4–8% |

The outliers appear abruptly at layer 2 and vanish after layer 20. Compression
difficulty is not uniform across depth.

### The outlier channels are the same regardless of input

`03_01` ran a second, unrelated prompt: **14–15 of the top 15 channels were
identical**. Which channels blow up is decided by the model's weights, not by
the text.

This is the result the whole design rests on. It is what makes it legitimate to
compute a channel ordering once, offline, ship it with the model, and reuse it
at runtime for free. Without it there would be no justification for
precomputing anything.

### Kurtosis predicts real damage, not just statistical oddity

`04` and `06` connected the statistic to consequences: storm layers show
roughly 9,000× worse quantization MSE than calm layers, and output KL of 5–11
against ~0.2 for calm layers. KL also proved more reliable than top-1 token
accuracy as the quality metric.

### Grouped beats uniform by 13–99× at equal size

`09_02`, comparing grouped(4,4,4) at 46,312 B against uniform 4-bit at
44,804 B — grouped is 3.4% *larger*:

| Layer | Grouped KL | Uniform KL | Ratio |
|---|---|---|---|
| 1 | 0.031 | 2.434 | 78× |
| 3 | 0.439 | 7.401 | 17× |
| 10 | 0.548 | 7.366 | 13× |
| 20 | 0.181 | 4.148 | 23× |
| 22 | 0.034 | 3.318 | 99× |

Quote the equal-size comparison. The "half the data of uniform 8-bit" version is
more dramatic but less rigorous.

### The end-to-end check

`validation/01` is the most direct evidence in the repository and needs no
understanding of KL divergence to read:

```
uniform 8-bit on storm layers   ->  Chinese-character garbage
grouped 4-bit, HALF the data    ->  perfect English
```

### How to spend bits

`05_01` swept the number of protected channels from 2 to 128 looking for an
elbow. **There isn't one** — error keeps dropping all the way out, with no
natural stopping point.

That matters more than it first appears. It means you cannot simply protect the
outliers and quantize the rest: the magnitude tails off gradually over hundreds
of channels, so every increase still buys accuracy and you would never stop
spending bytes. Stratifying into bands with different bit-widths is the answer,
not protection.

`05_04` found the rule — spend bits front-to-back, the shoulder needs 8 bits,
the bulk survives at 4 — but it holds only at loose budgets. Checked against
output KL at strict budgets, 13 of 23 layers need the bulk upgraded too. State
the rule with its condition.

### Why the split-point question could not be answered here

On Qwen, layer 1 sits *before* the storm. It is therefore both the cheapest
layer to compute and one of the easiest to compress — it wins on both axes at
once, and no optimizer can beat a layer that is best at everything.

The algebra is unambiguous. With total latency
`s·t·(slowdown − 1) + transfer(s) + const`, a later split *k* beats split 1 only
if `transfer(1) − transfer(k) > (k−1)·t·(slowdown − 1)`. The right side is
positive whenever the edge is slower than the cloud; the left side is at most
zero, because layer 1 already has the minimum byte cost at every quality
budget. Checked across 5 budgets × 3 bandwidths × 5 slowdowns — layer 1 won all
75 times.

So the split question is not merely harder on Qwen — it has no answer there.
The finding only became visible once the same measurements ran on Llama, where
layer 1 is the *most* outlier-concentrated layer rather than one of the calmest.

This is worth stating as a result rather than a limitation: activation
structure differs enough between models that a conclusion about split-point
selection drawn from a small model need not hold at scale, in either
direction.

---

## The Llama experiments

Numbered in the order they were run.

| Script | What it establishes |
|---|---|
| `01_activation_stats` | Layer 1 is the most outlier-concentrated layer (kurtosis 133,330; the top 1% of channels hold 64.7% of per-channel peak magnitude), declining smoothly with depth |
| `02_channel_cliff` | Group boundaries at channels 5 / 72 / 1312, placed where the sorted-magnitude curve changes character |
| `03_bytes_per_split` | Bytes needed per split point at each quality budget — superseded by `09` |
| `04_memory_ceiling` | Feasible split range from parameter counts: L6 on a busy handset, L10 when cleared |
| `05_robustness` | The split-point saving across 3 prompts × 3 sequence lengths |
| `06`, `06_01` | Quality calibration: what a KL budget means in perplexity and in readable output |
| `07`, `07_01` | Per-layer channel ordering versus one frozen ordering |
| `08_cross_text_order` | Whether orderings from calibration text work on unseen text |
| `09_bytes_per_split_perlayer` | Bytes per split, recomputed with per-layer orderings |
| `11a_collect` | Bulk measurement: 86,400 records across 300 WikiText passages |
| `11b`, `11c`, `12`, `13` | Whether any cheap statistic predicts per-input quality cost |
| `14_hard_prompts` | Domain generalisation — attempted twice, abandoned, see below |

---

## Results in detail

### Activation structure

Llama 3.1 8B's outliers are injected around layer 1 and ride the residual
connection unchanged: `max_abs` stays at 320–322 from layer 1 to layer 30 while
`max/mean` falls from 15,003 to 607. The outliers do not shrink — the ordinary
values grow around them.

This is the opposite shape from Qwen, where layers 0–1 are calm, layers 2–20
form a flat plateau, and 21–23 are calm again. That difference matters: on Qwen
layer 1 is both the cheapest to compute and one of the easiest to compress, so
it wins on both axes and no split optimization is possible. Qwen was
structurally the wrong model to test split selection on.

Consistent with Sun et al. 2024, *Massive Activations in Large Language
Models*.

### Compression

Channels ranked by magnitude, split into four bands, each quantized with its
own scale. A quantization scale is set by the largest value in its group, so
isolating the loud channels stops them coarsening the thousands of quiet ones.

At equal size, grouped compression gives 13–99× lower KL than per-tensor
uniform. The rule for spending bits — protect the front, the bulk survives at
4 bits — holds at loose quality budgets but breaks at strict ones, where more
than half the layers need the bulk upgraded too.

### Per-layer channel ordering

Measured across 300 texts:

```
layer 0, grouped(8,4,4):  frozen order KL 7.703  ->  own order 0.010   749x
layer 0, grouped(4,4,4):  frozen order 13.727    ->  own order 0.124   111x
```

Layer 0 keeps 0% of the frozen order's top-5 channels and misses 47 of its own
top 72 — the outliers have not been injected yet at that depth, so layer 2's
ranking is not a drifted version of layer 0's but a different population
entirely.

The benefit grows with distance from the calibration layer: 1.52× at L3, 2.68×
at L7, 4.12× at L11, tracking the drift in group membership (100% agreement at
L2 falling to 73% at L11).

Cross-text validation used orderings derived from a third domain entirely and
applied them to two others. They matched a same-text oracle's scheme selection
in 45/48 and 42/48 cases. The channels that matter are identical regardless of
text — `top72 kept` is 100% at every layer except 0 — which is why precomputing
works.

### Quality calibration

Reference perplexity for the uncompressed model, then the same text through
each scheme. Grouped(8,4,4) at layer 7 costs +0.27% perplexity on technical
prose and +3.01% on narrative. Uniform quantization at 4 and 8 bits produces
degenerate output at every split.

An important caution: perplexity and readability can diverge. A 56.6%
perplexity rise at layer 1 still produced fluent text — it simply said
different things. Do not treat "under 1% is imperceptible" as a law.

### Per-input variation, and why it cannot be detected

Quality cost varies about 8× across texts at a fixed layer and scheme
(L3, grouped(8,4,4): KL 0.035 to 0.273 across 300 texts). Five approaches to
predicting it, all measured **within** a fixed (layer, scheme, sequence length)
cell — pooling across layers merely recovers the already-known layer effect:

| Approach | Best result |
|---|---|
| 9 activation statistics, individually, Pearson | \|r\| = 0.391 |
| Same, monotone non-linear (Spearman) | \|r\| = 0.397 |
| Same, combined, held-out validated | mean R² = 0.000 |
| Token-position structure and effective rank | \|r\| = 0.446 |
| Random forest, regression and classification | held-out R² negative; AUC 0.533 |

Adding the activation features to a global model that already knows the layer
and scheme changes held-out R² by −0.002.

The distinction that emerged, and it took a while to see: **which channels are
loud** is input-independent, which is what makes the frozen ordering work.
**How much damage quantizing them causes** is not. Earlier experiments showed
identical *scheme selections* across prompt styles, which is compatible with 8×
variation in the underlying KL — scheme selection is discrete and buckets a
wide range into the same choice.

The claim, in the narrow form the evidence supports: *the tested low-cost
handcrafted activation statistics did not provide sufficient out-of-sample
predictive accuracy for reliable per-input hardness estimation, under either
linear or non-linear learned models. Richer representations and deployable
text-level difficulty features remain untested.*

---

## Corrections

Claims that were believed, then found wrong. They are documented rather than
deleted, because the failure modes recur.

**A 24% split-point saving, withdrawn.** Measured using one frozen channel
ordering. Later experiments showed per-layer orderings are better and should be
adopted — and under those, layer 0 stops being expensive and the saving is
0.00%. The two results could not both stand. What survives: layers 1–2, where
the massive activations are injected, cost 22.9% more than layer 3 onward at
strict budgets, with the mechanism intact.

**A quality table 3.7× optimistic.** Calibrated on a single text, which turned
out to be easier than *all 300* WikiText passages — its KL of 0.0221 sits below
the corpus minimum of 0.035, against a mean of 0.083. The controller was
reporting that it met a strict budget while missing it for most inputs. Fixed
by selecting on the 95th percentile.

**Split-point results that were measurement noise.** An early experiment
reported the optimal split moving with sequence length. In fact the byte cost
was identical at every split (one value repeated 23 times) and so was the
compute, so the "finding" was timing jitter — run-to-run noise was 6× larger
than the entire spread across splits.

**Per-layer timings measuring the wrong thing.** A 0.5B model cost 26.7 ms at 5
tokens and 27.7 ms at 250. Fifty times the work for 4% more time is kernel
launch overhead, not compute. Every latency claim resting on those numbers was
withdrawn, and no result in this repository depends on timing.

**A channel ordering the receiver could not reproduce.** The compression ranked
channels from the live activation, which the decoder has no access to, and the
cost of transmitting that ranking was never counted — about 6 KB per request.
Fixed by freezing the ordering and shipping it with the model, which turned an
accounting error into a design decision.

**Memory ceilings off by one.** The code computed a *count* of layers and
printed it as an *index*. Since split *k* means layers 0..k inclusive, k layers
fitting gives a deepest split of k−1. Every ceiling shifted down one.

**A domain-generalisation experiment that failed twice.** First attempt padded
short passages by repetition, so the model had already seen the tokens and
perplexity collapsed. Second attempt used long standalone passages — Hamlet,
Richard III, standard Wikipedia articles — which came out 5–8× *easier* than
WikiText because they are famous and heavily memorised. Selecting for fame, not
difficulty. Domain generalisation of the quality table remains untested, and is
stated as a limitation.

---

## Methodology notes

**Quality is measured as KL divergence averaged over all token positions**, not
just the last. An earlier version corrupted 100 positions and scored one.

**Correlations are measured within a fixed cell.** Pooling across layers
produces a stronger-looking number driven entirely by the layer effect, which
is already known. In this data the pooled figure (0.442) exceeds every
within-cell figure (max 0.391).

**Learned models are validated on held-out prompts, split by prompt rather than
by record.** With 8 predictors and 300 points a model fits noise readily:
in-sample R² averaged 0.115 while held-out averaged 0.000, and several cells
were negative.

**A noise floor of roughly ±0.5 percentage points** was established from
impossible measurements — several runs reported perplexity *below* the
uncompressed baseline, which cannot happen, so those negative values are the
measurement error made visible. Any gain under about 1pp is not distinguishable
from zero. One example: the same layer and scheme gave −0.67pp in one run and
+0.64pp in another, differing only in the text.

---

## Reproducing

```bash
python3 experiments_llama/01_activation_stats.py     # ~1 min
python3 experiments_llama/09_bytes_per_split_perlayer.py   # ~2 hrs
python3 experiments_llama/11a_collect.py             # ~70 min, 86,400 records
python3 experiments_llama/11b_analyse.py             # no GPU
```

The analysis scripts (`11b`, `11c`, `13`) read the saved JSON and need no GPU,
so results can be re-derived without repeating the measurement.

---

## Requirements

Measured and run on:

| | |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| transformers | 5.12.1 |
| numpy | 2.4.4 |
| datasets | 5.0.1 |
| matplotlib | 3.11.0 |
| GPU | NVIDIA A100-PCIE-40GB, driver 570.172.08 |

```bash
pip install torch transformers numpy datasets matplotlib
```

Versions matter here. Several APIs these scripts touch have changed
recently:

- `model.model.rotary_emb(...)` and `position_embeddings=` — the experiments
  call decoder layers directly rather than going through `model.forward()`, and
  that signature differs across Transformers versions
- `torch_dtype=` is deprecated in favour of `dtype=` in transformers 5.x; the
  code still uses the old name and emits a warning
- The WikiText dataset must be loaded as `Salesforce/wikitext`. The bare name
  `wikitext` no longer resolves in recent `huggingface_hub` — it requires the
  full `namespace/name` form

**Model access.** `meta-llama/Llama-3.1-8B-Instruct` is gated on Hugging Face.
Accept the licence on the model page, then `hf auth login`. The weights are
about 16 GB and download once. Qwen2.5-0.5B-Instruct is not gated.

**Storage.** The bulk collection (`11a_collect`) writes a 43 MB JSON. Together
with the model cache, budget about 20 GB.

**No GPU needed** for the analysis scripts — `11b_analyse`, `11c_multivariate`
and `13_ml_hardness` read the saved JSON, so results can be re-derived without
repeating the measurement.

---

## Scope

**Prefill only.** Decode — token-by-token generation with a KV cache — has a
different transmission profile and is not modelled.

**Weight quantization is out of scope.** It solves memory; this work addresses
bandwidth. Under weight-only quantization the activation remains bf16, so the
transmitted tensor is unaffected by weight precision.

**Llama 3.1 8B only** at scale. 70B does not fit the available GPU.

**One calibration corpus.** WikiText-2, English encyclopaedic prose.
