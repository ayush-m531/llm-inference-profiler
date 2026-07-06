# LLM Inference Profiler

Profiling and optimization for **edge-cloud split inference** of large language models,
with a focus on **quantization-aware split-point selection**.

This is the experimental codebase for an M.Tech thesis:
*"Compute-Communication Co-optimization for Edge-Cloud LLM Inference: Network-Aware Layer
Partitioning with Quantization-Aware Split Points."*

Model used throughout: **Qwen2.5-0.5B-Instruct**. Experiments run on an NVIDIA A100 GPU.

---

## Motivation

Large language models are usually run entirely in the cloud. For many real applications,
this is not ideal:

- **Privacy:** Sending raw user input to the cloud means private data (messages, medical
  or legal text, photos) leaves the device. For privacy-sensitive applications, and under
  regulations like GDPR and HIPAA, this is often not acceptable. Running the early layers
  on-device keeps the raw input local — only an intermediate representation is transmitted,
  which is much harder to invert back to the original input.
- **Availability:** A fully cloud-based system stops working without connectivity (weak
  signal, offline, outages). A split system can do useful work locally and reach the cloud
  only when needed.
- **Cost:** Serving a large model in the cloud for many users is expensive, since compute
  is paid per request. Offloading part of the computation to the user's device reduces
  server load and cost.

At the same time, running the full model entirely on the device is not feasible either —
large models exceed edge-device memory and are slow and power-hungry to run locally.

This forces a middle ground: split the model across the edge device and the cloud. The
research question is then **where to place the split, and how much to compress the data
transmitted at that split**, so that the system achieves low latency and low bandwidth
usage without significantly degrading model quality.

This project studies that trade-off, with a focus on how per-layer quantization
sensitivity should inform the joint choice of split point and quantization precision —
extending fixed-precision split-inference approaches such as EdgeShard.

---

## Repository structure

```
experiments/
├── quantization/       # how quantizable each layer is, and why
│   ├── 03_activation_stats.py       # per-layer activation statistics (kurtosis, outliers)
│   ├── 03_01_activation_stats.py    # same, second prompt (input-independence check)
│   ├── 04_quantization_error.py     # real quantization error (MSE) per layer, per bit-width
│   ├── 05_protected_quant.py        # keep top outlier channels in full precision
│   ├── 05_01_k_sweep.py             # how many channels to protect (error vs K)
│   ├── 06_endtoend_validation.py    # does quantizing a layer hurt the final output? (KL)
│   └── results/
│
└── split_network/      # where to split, and the network / joint decision
    ├── 01_layer_latency.py          # per-layer compute latency
    ├── 02_split_point_analysis.py   # compute cost of each split point
    ├── 07_network_split.py          # split cost with network transfer added
    ├── 08_quant_aware_split.py      # split + quantization, latency-only (baseline)
    ├── 08_01_controller.py          # split + quantization with a quality constraint
    └── results/
```

File naming: `NN_name.py` is a base experiment; `NN_MM_name.py` is a direct follow-up or
extension of experiment `NN`. Each experiment writes its outputs (JSON + plot) into the
matching `results/` folder.

---

## Key findings so far

- **Per-layer compute is uniform.** All 24 layers take roughly the same time; no single
  layer is a compute bottleneck.
- **Quantization sensitivity is highly non-uniform.** Middle layers (roughly 2–20) are far
  more sensitive to quantization than the first and last few layers — a sharp three-zone
  pattern, driven by a small, fixed set of outlier channels that is stable across inputs.
  This is consistent with the "massive activations" phenomenon reported in the literature.
- **The sensitivity signal is validated end-to-end.** Cheap per-layer signals (kurtosis)
  predict real quantization error, which in turn predicts actual degradation of the model's
  output.
- **Outlier channels can be protected cheaply.** Keeping a small number of channels in full
  precision substantially reduces quantization error in the sensitive layers.
- **Split point only becomes meaningful with variable-size transmission.** With fixed
  precision the transmitted activation is the same size at every split, so network cost is
  a constant offset. Quantization makes the transmitted size depend on the split, which is
  what turns split-point selection into a real optimization problem.

---

## Status

Active research code. Experiments and findings are evolving; results in this repository
reflect work in progress rather than final thesis results.
