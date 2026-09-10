# Results

Every measured number in this project, with the experiment that produced it.

Two models. **Llama-3.1-8B-Instruct** (32 layers, hidden 4096) is the target;
**Qwen2.5-0.5B-Instruct** (24 layers, hidden 896) is where the work began,
before GPU access. Llama measurements on an NVIDIA A100-40GB.

**Convention:** layers are indexed 0–31. "Split at layer *k*" means the edge
computes layers 0 through *k* inclusive — that is, *k*+1 layers — and transmits
the resulting activation. Compression is applied after layer *k* runs.

---

## Contents

- [Activation structure](#activation-structure)
- [Compression](#compression)
- [Channel ordering](#channel-ordering)
- [Quality calibration](#quality-calibration)
- [Split point selection](#split-point-selection)
- [Memory ceiling](#memory-ceiling)
- [Per-input variation](#per-input-variation)
- [The controller](#the-controller)
- [The two-process system](#the-two-process-system)
- [Corrections](#corrections)
- [Methodology](#methodology)
- [Future work](#future-work)
- [Limitations](#limitations)

---

## Activation structure

### Qwen: three zones
`experiments/quantization/03_activation_stats`

| Layers | Kurtosis | Top 1% of channels hold |
|---|---|---|
| 0–1 | 31–57 | ~5% |
| 2–20 | 68,000–84,000 | 47–50% |
| 21–23 | 88–258 | 4–8% |

Outliers appear abruptly at layer 2 and vanish after layer 20.

### Llama: a peak at layer 1, then decline
`experiments_llama/01_activation_stats`

| Layer | Kurtosis | Top 1% hold | max_abs |
|---|---|---|---|
| 0 | 4,122 | 28.0% | 3.2 |
| 1 | 133,330 | **64.7%** | 320.0 |
| 8 | 129,135 | 50.5% | 320.0 |
| 16 | 118,628 | 39.7% | 322.0 |
| 30 | 38,601 | 14.9% | 320.0 |
| 31 | 14 | 3.9% | 37.8 |

Layer 1 is the most outlier-concentrated layer in the model — the opposite of
Qwen, where layers 0–1 are the calmest.

**Mechanism.** `max_abs` stays at 320–322 from layer 1 to layer 30 while
`max/mean` falls from 15,003 to 607. The outliers do not shrink; the ordinary
values grow around them. They are injected around layer 1 and ride the residual
connection unchanged.

Consistent with Sun et al. 2024, *Massive Activations in Large Language
Models*.

> **Naming note.** "Top 1% hold" is the share of summed per-channel *peaks*
> held by the loudest 1% of channels — not the share of total magnitude.
> Reproducing it as the latter gives a different number.

### Outlier channels do not depend on the input
`experiments/quantization/03_01`

Running an unrelated second prompt on Qwen: **14–15 of the top 15 channels
identical**. Which channels blow up is decided by the model's weights, not by
the text.

This is the result the whole design rests on. It is what makes it legitimate to
compute a channel ordering once, offline, ship it with the model, and reuse it
at runtime for free.

---

## Compression

Channels are ranked by magnitude and split into four bands, each quantized with
its own scale:

| Group | Channels (Llama) | Typical bits |
|---|---|---|
| top | 5 | 16 (unquantized) |
| shoulder | 6–72 | 8 |
| mid | 73–1312 | 4 or 8 |
| bulk | 1313–4096 | 4 |

Per-group scales are the point. A quantization scale is set by the largest
value in its group, so isolating the loud channels stops them coarsening the
thousands of quiet ones.

Boundaries come from `experiments_llama/02_channel_cliff`, placed where the
sorted-magnitude curve changes character. Retuning them against Qwen's 10/40/150
saves about 1 bit at layer 1 and 0.6 bits at layer 30 — real but small.

### Grouped beats uniform by 13–99× at equal size
`experiments/split_network/09_02`

grouped(4,4,4) at 46,312 B against uniform 4-bit at 44,804 B — grouped is 3.4%
*larger*:

| Layer | Grouped KL | Uniform KL | Ratio |
|---|---|---|---|
| 1 | 0.031 | 2.434 | 78× |
| 3 | 0.439 | 7.401 | 17× |
| 10 | 0.548 | 7.366 | 13× |
| 20 | 0.181 | 4.148 | 23× |
| 22 | 0.034 | 3.318 | 99× |

### Uniform quantization is unusable, not merely worse
`experiments/validation/01`, `experiments_llama/06`

On Qwen:

```
uniform 8-bit on storm layers   ->  Chinese-character garbage
grouped 4-bit, HALF the data    ->  perfect English
```

On Llama, uniform quantization at 4 and 8 bits gives perplexity rises of 10⁵ to
10⁷ at every split tested — output like `">>> ; ; ; ;"` and
`"went went went with with with"`.

There is no equal-quality point at which to compare bytes. The comparison is a
feasibility difference, not a compression-ratio improvement.

> **Known weakness.** The baseline is per-tensor uniform quantization only.
> Its failure is arithmetically predictable: at Llama layer 1, `max_abs` is
> 320.0 and `mean_abs` is 0.0213, so an int8 scale of 320/127 = 2.52 rounds
> anything below 1.26 to zero — 59× above the mean. Per-token and per-channel
> scaling are the fair comparisons and are **untested**. Per-token int4 costs
> 205,200 B at 100 tokens against grouped(4,4,4)'s 205,562 B, so it is cheaper
> than the cheapest scheme measured here.

### How to spend bits
`experiments/quantization/05_01`, `05_04`

Sweeping the number of protected channels from 2 to 128 finds **no elbow** —
error keeps dropping with no natural stopping point.

That matters more than it appears. It means you cannot simply protect the
outliers and quantize the rest: the magnitude tails off gradually over hundreds
of channels, so every increase still buys accuracy and you would never stop
spending bytes. Stratifying into bands is the answer, not protection.

The rule — spend bits front to back, shoulder needs 8 bits, bulk survives at 4
— holds at loose budgets. At strict budgets 13 of 23 layers need the bulk
upgraded too. State the rule with its condition.

---

## Channel ordering

### Per-layer orderings cut error by up to 749×
`experiments_llama/07`, `07_01` — 300 texts, seq 256

Mean KL, one frozen ordering (from layer 2) versus each layer's own:

| Layer | Scheme | Frozen | Per-layer | Ratio |
|---|---|---|---|---|
| 0 | grouped(8,4,4) | 7.703 | 0.010 | **749×** |
| 0 | grouped(4,4,4) | 13.727 | 0.124 | 111× |
| 0 | grouped(8,8,4) | 0.738 | 0.003 | 283× |

Layer 0 keeps **0%** of the frozen ordering's top-5 channels and misses 47 of
its own top 72. The outliers have not been injected at that depth, so layer 2's
ranking is not a drifted version of layer 0's but a different population
entirely.

**The benefit grows with distance from the calibration layer**, tracking the
drift in group membership:

| Layer | Frozen KL | Per-layer KL | Ratio | Same-group agreement |
|---|---|---|---|---|
| 3 | 0.00687 | 0.00450 | 1.52× | 94% |
| 5 | 0.00543 | 0.00258 | 2.11× | 86% |
| 7 | 0.00521 | 0.00195 | 2.68× | 80% |
| 9 | 0.00621 | 0.00182 | 3.41× | 75% |
| 11 | 0.00701 | 0.00170 | 4.12× | 73% |

### Orderings from calibration text work on unseen text
`experiments_llama/08_cross_text_order`

Orderings derived from a third domain entirely, applied to two others:

| Evaluation text | Matched same-text oracle | Changed scheme vs frozen | Mean saving |
|---|---|---|---|
| technical | 45/48 (94%) | 15/48 | 28.9% |
| narrative | 42/48 (88%) | 21/48 | 27.7% |

**Why it works:** `top72 kept` is 100% at every layer except 0. The channels
that matter are identical between calibration and evaluation text; the ~20%
disagreement at layer 11 sits entirely in the mid and bulk groups, where
misplacement costs almost nothing.

Calibration orderings are not a degraded oracle — 07 with oracle orderings gave
14/48 changes on the technical text, 08 with calibration orderings gave 15/48.

**Why this matters for deployment.** Transmitting the ordering per request
would cost 4096 indices at 12 bits — about 6 KB for a list that never changes,
roughly 1% overhead at 256 tokens but 60% at 5 tokens. Precomputing it costs
12 layers × 4096 channels × 2 bytes ≈ 98 KB against a 14 GB model.

---

## Quality calibration

`experiments_llama/06`, `06_01` — what a KL budget means in perplexity and in
readable output

Reference perplexity of the uncompressed model: **2.56** on technical prose,
**5.55** on narrative. Perplexity measures how hard the *text* is to predict —
never compare raw values across texts, only rises against each text's own
baseline.

Scheme needed at a 1% perplexity budget, as a fraction of bf16:

| Split | Technical | Narrative |
|---|---|---|
| L1 | 33.1% | 33.1% |
| L7 | 25.5% | 33.1% |
| L11 | 50.1% | 50.1% |

**Held across both texts:** L7 needs a cheaper scheme than L11; L1 is worst;
uniform is unusable everywhere.

**Did not hold:** L7's absolute cost. Harder text leaves less headroom. Report
the range, not a point.

**Surprise:** the byte cost is flat across layers 5–26, but perplexity is not.
Quality varies inside the byte plateau. No mechanism established.

> **Caution.** Perplexity and readability diverge. A 56.6% perplexity rise at
> layer 1 still produced fluent text — it simply said different things. Do not
> treat "under 1% is imperceptible" as a law.

---

## Split point selection

### The 24% saving, withdrawn
`experiments_llama/03_bytes_per_split` → superseded by `09`

Exp 03 measured a 24% byte saving for splitting in the middle rather than at
layer 0/1, using one frozen channel ordering. Exp 07/08 then showed per-layer
orderings are better and should be adopted. Under those, layer 0 stops being
expensive.

Re-measured directly, both orderings side by side:

| Budget | Frozen ordering | Per-layer orderings |
|---|---|---|
| KL ≤ 0.05 | 22.89% | **−1.63%** |
| KL ≤ 0.10 | 24.12% | **0.00%** |
| KL ≤ 0.25 | 1.60% | **0.00%** |

The 24% was an artifact of a suboptimal ordering.

**And it is worse than it looks.** If layer 0 were the optimal split there
would be no reason to split at all — a layer-0 activation costs ~205 KB per 100
tokens against ~300 bytes for the raw token IDs, about 680× cheaper. So bytes
cannot select the split point.

### What survives

Under per-layer orderings at a strict budget, layers 1–2 — exactly where the
massive activations are injected — cost **22.9% more** than layer 3 onward.

Two caveats, both stated: the comparison basis (L1/L2 vs L3+) was chosen after
seeing the data, and the penalty exists only at strict budgets.

---

## Memory ceiling

`experiments_llama/04_memory_ceiling` — from parameter counts, no GPU

| | Per layer | Embedding | Whole model |
|---|---|---|---|
| bf16 | 416.0 MB | 1002.0 MB | 13.98 GB |
| int8 | 208.0 MB | 501.0 MB | 6.99 GB |
| int4 | 104.0 MB | 250.5 MB | 3.49 GB |

218,112,000 parameters per layer; 525,336,576 in the embedding. Llama 3.1 uses
grouped-query attention — 32 query heads but 8 KV heads — so k and v
projections are a quarter of q's size. Assuming otherwise overcounts and every
ceiling comes out wrong.

Deepest feasible split, assuming 70% of free RAM available for weights:

| Free RAM | bf16 | int8 | int4 | |
|---|---|---|---|---|
| 2.00 GB | L0 | L3 | L10 | |
| 4.00 GB | L3 | L10 | L24 | |
| **5.90 GB** | **L6** | L16 | ALL | measured: phone under normal use |
| **7.81 GB** | **L10** | L23 | ALL | measured: same phone, apps cleared |

The two measured values come from a six-year-old handset with 12 GB installed.
Same device, ceiling four layers apart depending on what else is running.

> **Assumption.** The 70% figure is a judgement call, not a measurement — the
> remainder covers activations, KV cache, runtime and OS headroom. The deep end
> is sensitive to it: 0.67 gives a ceiling one layer shallower than 0.70.

---

## Per-input variation

`experiments_llama/11a` — 86,400 records across 300 WikiText-2 passages, 3
sequence lengths, 12 layers, 4 schemes, 2 orderings. 70.9 minutes on an A100.

### Single-text calibration was 3.7× optimistic

The controller's original quality table said L3 with grouped(8,4,4) gives KL
0.0221. Across 300 texts:

| | Value |
|---|---|
| mean | 0.0828 |
| median | 0.0769 |
| p95 | 0.1411 |
| min | 0.0347 |
| max | 0.2733 |

The calibration text was easier than **all 300** — its 0.0221 sits below the
corpus minimum. Under a strict budget of 0.05 the controller judged the scheme
safe while the real distribution is centred on 0.083.

Fixed by selecting on the 95th percentile.

### Quality cost varies ~8× across inputs

At a fixed layer, scheme and sequence length: L3 with grouped(8,4,4) ranges
0.035 to 0.273 across 300 texts.

### Sequence length barely matters

Mean KL at seq 512 divided by mean KL at seq 128, grouped(8,4,4):

| Layer | 128 | 256 | 512 | Ratio |
|---|---|---|---|---|
| 0 | 0.01084 | 0.01029 | 0.00938 | 0.86 |
| 1 | 0.38814 | 0.44334 | 0.51858 | 1.34 |
| 3 | 0.08804 | 0.08283 | 0.07837 | 0.89 |
| 11 | 0.00994 | 0.00984 | 0.00945 | 0.95 |

Ratios cluster 0.85–1.2 across all schemes and layers. A 4× change in length
moves KL by at most about 20%, so the lookup table needs no length dimension.

*(Bytes scale exactly with length. That is arithmetic, not a finding.)*

### Nothing predicts which inputs are hard
`experiments_llama/11b`, `11c`, `12`, `13`

There is real variation to detect. Five approaches, all measured **within** a
fixed (layer, scheme, sequence length) cell — pooling across layers merely
recovers the already-known layer effect:

| Approach | Best result |
|---|---|
| 9 activation statistics, individually, Pearson | \|r\| = 0.391 |
| Same, monotone non-linear (Spearman) | \|r\| = 0.397 |
| Same, combined, held-out validated | mean R² = 0.000 |
| Token-position structure, effective rank | \|r\| = 0.446 |
| Random forest, regression and classification | held-out R² negative; AUC 0.533 |

Adding the activation features to a global model that already knows the layer
and scheme changes held-out R² by **−0.002**. Layer and scheme alone explain
96%.

The random forest *overfits*: in-sample R² 0.472, held-out −0.046. It performs
worse than predicting the mean.

**The distinction that emerged**, and it took a while to see:

- **Which channels are loud** — input-independent. This is what makes the
  precomputed ordering work.
- **How much damage quantizing them causes** — varies ~8× by input.

Earlier experiments showed identical *scheme selections* across prompt styles,
which is compatible with 8× variation underneath: scheme selection is discrete
and buckets a wide range into the same choice.

**The claim, in the narrow form the evidence supports:** the tested low-cost
handcrafted activation statistics did not provide sufficient out-of-sample
predictive accuracy for reliable per-input hardness estimation, under either
linear or non-linear learned models. Richer activation representations and
deployable text-level difficulty features remain untested.

---

## The controller

Lives in the [companion repository](https://github.com/ayush-m531/edge-split-controller).

### Memory pressure changes the answer

Same handset, minutes apart, strict budget:

| Free RAM | Cap | Split | Scheme | Bytes | % of bf16 |
|---|---|---|---|---|---|
| 5.90 GB (busy) | L5 | L1 | grouped(8,8,4) | 270,912 | 33.1% |
| 7.81 GB (cleared) | L9 | L6 | grouped(8,4,4) | 208,912 | 25.5% |

23% fewer bytes because the user closed some applications. At 5.90 GB the
reachable layers are L1–L5 and none meets a strict budget with the cheaper
scheme (L5's p95 is 0.054, just over 0.05), so it falls back.

### Only two configurations exist at a strict budget

Verified by sweeping free RAM from 2 GB to 14 GB:

| Free RAM | Decision |
|---|---|
| 2–3 GB | cloud only |
| 4–5.9 GB | L1 grouped(8,8,4), 270,912 B |
| 6.4–14 GB | L6 grouped(8,4,4), 208,912 B |

Layers 1–5 all require grouped(8,8,4); L6 is the first where grouped(8,4,4)
qualifies. Since the search takes the shallowest layer achieving the cheapest
cost, L2–L5 are never chosen — same cost as L1, more edge work.

This is a structural property, not a thin scenario. The variety lives in the
*budget*: strict, balanced and relaxed give L6, L4 and L8 at identical
conditions.

### The split search earns its complexity

"Layers saved" is edge compute avoided by taking the shallowest layer with the
same transmitted size:

| Free RAM | Quality | Split | Deepest feasible | Saved |
|---|---|---|---|---|
| 5.90 GB | strict | L1 | L5 | 4 |
| 5.90 GB | balanced | L4 | L5 | 1 |
| 7.81 GB | strict | L6 | L9 | 3 |
| 7.81 GB | balanced | L4 | L9 | 5 |
| 7.81 GB | relaxed | L8 | L9 | 1 |

An earlier version went as deep as memory allowed and would have paid the
"deepest" column every time.

### Hysteresis

Over a 21-step scenario: **8 reconfigurations, not 21**. Each one means the
edge loading or dropping layer weights at 416 MB per layer.

The controller compares *outcomes*, not causes. At step 8 the temperature
crosses 80°C and the binding constraint changes from memory to thermal — but
memory had already forced L1, so the decision is unchanged and nothing
reconfigures. An implementation reacting to "thermal limit crossed" would
reconfigure for nothing.

---

## The two-process system

`edge.py` / `cloud.py` in the companion repository. Five cases, run end to end.
"Socket bytes" is what actually crossed the TCP connection.

| Case | Split | Scheme | Socket bytes | % of bf16 |
|---|---|---|---|---|
| Normal, 7.81 GB | L6 | grouped(8,4,4) | 25,068 | 25.5% |
| Memory pressure, 5.90 GB | L1 | grouped(8,8,4) | 32,508 | 33.1% |
| Relaxed budget | L8 | grouped(4,4,4) | 24,666 | 25.1% |
| Mid-pass abort at L3 | L3 | grouped(8,8,4) | 32,508 | 33.1% |
| 2 GB free | — | cloud only | — | — |

**The byte model is accurate:** 25,068 measured against 25,080 predicted, a
12-byte gap accounted for by header scales versus the table's assumed 4 bytes
per group.

### Mid-pass abort

```
planned split     L6, grouped(8,4,4)
aborted at        L3
scheme re-chosen  grouped(8,4,4) -> grouped(8,8,4)
bytes             25,068 -> 32,508  (+30%)
cloud resumed     from L4, ran 28 layers instead of 25
output            unchanged and correct
```

Three things had to work. The split index travels with the payload, so the
cloud resumes correctly — a fixed-split system would have started at L7 and
silently skipped layers 4, 5 and 6. The scheme is re-chosen, because
grouped(8,4,4) has a p95 of 0.141 at L3, well over the 0.05 budget. And the 30%
penalty is the correct trade against overheating.

---

## Corrections

Claims that were believed, then found wrong. Documented rather than deleted,
because the failure modes recur.

### C1. A 24% split-point saving, withdrawn

Measured with one frozen channel ordering. Later work showed per-layer
orderings are better — and under those the saving is 0.00%. Two experiments in
the same repository contradicted each other; the conflict was found by external
review, not by me. See [Split point selection](#split-point-selection).

### C2. A quality table 3.7x optimistic

Calibrated on a single text that turned out to be easier than all 300 WikiText
passages. The controller reported meeting a strict budget while missing it for
most inputs. See [Per-input variation](#per-input-variation).

### C3. Split-point results that were measurement noise

An early experiment reported the optimal split moving with sequence length. In
fact the byte cost was identical at every split — one value repeated 23 times —
and so was the compute, so the "finding" was timing jitter. Run-to-run noise was
**6× larger** than the entire spread across splits.

### C4. Per-layer timings measuring the wrong thing

A 0.5B model cost 26.7 ms at 5 tokens and 27.7 ms at 250. Fifty times the work
for 4% more time is kernel launch overhead, not compute. Every latency claim
resting on those numbers was withdrawn, and **no result in this repository
depends on timing**.

### C5. A channel ordering the receiver could not reproduce

The compression ranked channels from the live activation, which the decoder has
no access to, and the cost of transmitting that ranking — about 6 KB per
request — was never counted. Fixed by freezing the ordering and shipping it
with the model, which turned an accounting error into a design decision.

### C6. Memory ceilings off by one

The code computed a *count* of layers and printed it as an *index*. Since split
*k* means layers 0..*k* inclusive, *k* layers fitting gives a deepest split of
*k*−1. Every ceiling shifted down one; L7/L11 became L6/L10.

### C7. A domain-generalisation experiment that failed twice

First attempt padded short passages by repetition, so the model had already
seen the tokens and perplexity collapsed to ~1.3 against WikiText's 11.5.

Second attempt used long standalone passages with a length guard — Hamlet,
Richard III, standard Wikipedia articles, Hindi and Gujarati prose. They came
out **5–8× easier** than WikiText, because famous text is heavily memorised.
Selecting for fame, not difficulty. WikiText itself spans 3.99 to 38.09
reference perplexity; every hand-picked passage fell below its minimum.

Domain generalisation of the quality table remains untested.

### C8. Two bugs in the controller

The safety margin was applied *after* the device caps, so low battery capped
the split at L1 then subtracted 1, landing on L0 — which is excluded — and
falsely reported no feasible split. The margin belongs only on the memory
ceiling, where the 70% assumption creates real uncertainty.

Separately, the printed cap and the value `decide()` actually used disagreed:
the table showed L5 at 5.90 GB while the function returned a split of L6, above
its own stated cap. Caught by checking the two against each other rather than
trusting either.

---

## Methodology

**M1. Quality is KL divergence averaged over all token positions**, not just the
last. An earlier version corrupted 100 positions and scored one.

**M2. Correlations are measured within a fixed cell.** Pooling across layers gives a
stronger-looking number driven entirely by the layer effect. In this data the
pooled figure (0.442) exceeds every within-cell figure (max 0.391).

**M3. Learned models are validated on held-out prompts, split by prompt rather than
by record.** With 8 predictors and 300 points a model fits noise readily:
in-sample R² averaged 0.115 while held-out averaged 0.000, several cells
negative. Splitting by record would leak, since each prompt contributes 48 rows
in a global model.

**M4. A noise floor of roughly ±0.5 percentage points** was established from
impossible measurements: several runs reported perplexity *below* the
uncompressed baseline, which cannot happen, so those negatives are the
measurement error made visible. Any gain under about 1pp is not distinguishable
from zero.

One demonstration — the same layer and scheme, differing only in the text:

```
run A:  gain -0.67pp   (per-layer ordering looked worse)
run B:  gain +0.64pp   (per-layer ordering looked better)
```

**M5. Discrete outcomes hide continuous variation.** Scheme selection buckets a wide
KL range into the same choice, so identical decisions across prompts are
compatible with 8× variation underneath. Mistaking one for the other cost
several days.

---

## Future work

In rough order of what would change the most.

### 1. Time as a second objective

Everything here optimises **bytes**. The controller minimises transmitted size
subject to a quality budget and ignores latency entirely, because the only
timing available was unreliable (see below).

With trustworthy timing the controller could trade compute against
transmission: is it faster to run three more layers on the device and send
less, or to send more and let the cloud do the work? That is the optimization
the project's title promises and it is currently unanswerable.

It also opens an inversion worth testing: **when bandwidth is plentiful,
compression may not be worth doing at all.** Sending bf16 uncompressed costs
more bytes but zero quantization time and zero quality loss. On a fast link
that could be the better choice. The controller would then adapt along two axes
— bytes when the network is constrained, latency when it is not — rather than
assuming compression is always correct.

### 2. Trustworthy edge timing

Prerequisite for the above.

The timings inherited from the early phase measured GPU kernel *launch
overhead*, not compute: a 0.5B model cost 26.7 ms at 5 tokens and 27.7 ms at
250. Fifty times the arithmetic for four percent more time is dispatch cost
dominating a small eager-mode model. Multiplying those figures by a "slowdown
factor" to simulate a weak device produced a meaningless number, and it erred
in the direction that flattered the thesis — real compute grows with tokens, so
transfer would dominate *less* at long sequences, not more.

Doing it properly needs three things: measurement on real hardware rather than
an A100 with an invented multiplier; correct methodology (warmup discarded,
synchronisation in the right places, and a sanity check that time scales with
token count); and a stated noise floor, since the original failure was
compounded by run-to-run variance exceeding the differences being compared.

### 3. Decode without per-token transmission

All measurements here cover prefill. Generation was left unmodelled because
per-token transmission is a different regime — many small messages,
latency-bound rather than bandwidth-bound.

But the cloud already holds every layer. So once prefill is done it can run
generation entirely on its own and stream the tokens back, with no activation
crossing the network per token and no round trip per step. That removes the
problem rather than solving it, and it is what a deployed system would do. The
edge's work ends when the prompt has been processed.

### 4. A mobile edge client

Device conditions — free memory, battery, temperature — are currently passed as
command-line flags. The two memory figures used throughout (5.90 and 7.81 GB)
are real measurements from a handset, but taken by hand.

A small phone application reading those values live, with a quality selector
for the per-request budget, would make the controller's inputs real rather than
simulated. It would also supply genuine thermal and battery *response* — how
fast the device actually heats under sustained inference — which is what the
mid-pass abort trigger needs. The abort policy is implemented and works; only
the trigger is a flag.

### 5. Per-token and per-channel compression baselines

The compression is compared against **per-tensor uniform quantization**, which
is a weak opponent: its failure is arithmetically predictable, since at Llama
layer 1 an int8 scale of 320/127 rounds anything below 1.26 to zero and the
mean absolute value is 0.0213.

The method here is **group-wise along the channel dimension** — channels ranked
by magnitude, split into four bands, each band sharing one scale. Two fairer
comparisons were never run:

| Scheme | Scales | Bytes at 100 tokens |
|---|---|---|
| per-tensor uniform | 1 | 204,804 |
| **this work: 4 magnitude bands** | 4 | 205,562 |
| per-token int4 | one per token | 205,200 |
| per-channel int4 | one per channel | 212,992 |

Per-channel is the fine-grained version of what this method approximates. If it
does as well with simpler machinery, the banding does not earn its complexity.

Per-token slices the other axis, and Sun et al. report that massive activations
concentrate at specific token *positions* as well as channels — so isolating
the damage to a few tokens may work better than isolating it to a few channels.
It is also **cheaper than the cheapest scheme measured here**.

Either outcome is useful: a win confirms the banding, a loss identifies where
the structure actually lives.

### 6. Runtime hardness prediction

Per-input quality cost varies about 8× at a fixed layer and scheme, and nothing
cheap predicts it. Five approaches were tested and all failed — see
[Per-input variation](#per-input-variation). The conclusion is narrow and
deliberately so: *the tested low-cost handcrafted activation statistics* do not
predict it.

What was **not** tested, and could:

- **Richer activation representations.** Everything tried was a scalar summary.
  PCA over the full tensor, or a learned embedding, might find structure that
  kurtosis and concentration measures cannot. The catch is deployability — a
  4096-dimensional projection is not a microsecond-cost runtime statistic.
- **Text-level difficulty features.** Every statistic tested came from the
  *activation*. None came from the *text*. Reference perplexity was the
  strongest predictor at 0.391 but is not deployable, since computing it needs
  the uncompressed model and the true next tokens. Cheap approximations — token
  rarity from a frequency table, type-token ratio, punctuation density, or a
  small auxiliary model's perplexity — are a genuinely different axis and are
  untested.
- **Deeper layers.** The positional correlation strengthened monotonically with
  depth: 0.190 at L3, 0.362 at L6, 0.446 at L9, in every positional predictor.
  Three points is thin, but it is monotone. Whether it continues past L11 was
  not tested, because those layers fall outside the memory-feasible range —
  though a positive result there would still be informative.

If a predictor were found, the controller could compress aggressively on easy
inputs and conservatively on hard ones. The current p95 approach is blunter: it
provisions for the hard case regardless, so easy inputs pay more bytes than
they need.

### 7. A reconfiguration cost model

The controller suppresses reconfiguration when the decision is unchanged, which
takes a 21-step scenario from 21 reconfigurations to 8. But it has no explicit
model of what reconfiguring *costs*.

Changing the split means the edge loading or dropping layer weights at 416 MB
per layer, against a transmission saving of roughly 62 KB per request. On those
numbers, reconfiguring is only worthwhile if conditions stay stable for a large
number of subsequent requests. A dwell time, or an explicit cost model
comparing reconfiguration against expected future saving, would make that
trade-off explicit rather than implicit in the hysteresis.

This would also address the oscillation weakness: hysteresis currently acts on
*decisions*, so genuine alternation between two valid answers would still
thrash.

### 8. Replication at 70B

Llama 3.1 70B does not fit a 40 GB GPU. Whether the activation structure holds
at that scale is unknown — and Qwen versus Llama already showed that structure
differs enough between models to reverse a conclusion, so it should not be
assumed in either direction.

Worth doing if larger hardware becomes available.

---

## Limitations

- **Prefill only.** Decode — token-by-token generation with a KV cache — has a
different transmission profile: many small messages rather than one large one,
latency-bound rather than bandwidth-bound. Not modelled.

- **The compression baseline is per-tensor uniform only.** Per-token and
per-channel scaling are the fair comparisons and are untested. Per-token int4
is cheaper than the cheapest scheme measured here.

- **One calibration corpus.** WikiText-2, English encyclopaedic prose. Domain
generalisation untested — two attempts failed for the reasons above.

- **p95 is a probabilistic guarantee.** The worst 5% of inputs still exceed the
budget. Nothing reaches 100%; the next text could be worse than anything
measured.

- **Timing is not modelled.** Available per-layer timings measured kernel launch
overhead. Edge compute cost does not enter the controller's decision.

- **The controller's hysteresis acts on decisions, not inputs.** It suppresses a
repeat of the same answer but would not suppress genuine oscillation between two
valid answers. Free memory hovering across a layer-fit boundary would thrash.

- **Weight quantization is out of scope.** It solves memory; this work addresses
bandwidth. Under weight-only quantization the activation remains bf16, so the
transmitted tensor is unaffected by weight precision.

- **Llama 3.1 8B only** at scale. 70B does not fit the available GPU.
