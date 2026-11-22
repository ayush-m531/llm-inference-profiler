# Compute-Communication Co-optimization for Edge-Cloud LLM Inference

M.Tech thesis, IIT Gandhinagar · Guide: Prof. Anirban Dasgupta · Jan 2026 – present

## Problem

A large language model like Llama 3.1 8B is too big to run fully on a small device. One option is to split it: the device runs the first part of the model and a cloud server runs the rest. The intermediate data passed between the two parts can be large. This thesis studies how much that data can be compressed without hurting the quality of the model's output.

## Setup

All experiments ran on a single NVIDIA A100 GPU (GRACE server). The device–server split is simulated on that one machine. The reported data size is the size of the compressed data that *would* be sent, not a measurement over a real network.

## Results

| Setting | Data to send (vs. original size) | Quality loss |
|---|---|---|
| Higher quality | 33.1% | under 1% for 95% of 300 test passages |
| Smaller size | 25.5% | 1.9% median |

- Ran 86,400 test cases (300 passages × 288 settings) in 70.9 minutes with an unattended Python/PyTorch test harness.
- Quality loss is reported across all 300 passages (95th percentile and median), not just the average, because it differs a lot from one input to another.

## Code

The code is private until the paper is published. I'm happy to walk through the work in an interview.
