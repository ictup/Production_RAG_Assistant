---
title: "FlashAttention Notes"
source_type: "markdown"
topic: "attention"
difficulty: "advanced"
tags: ["attention", "gpu", "memory"]
workspace_id: "public"
visibility: "public"
---

# FlashAttention

FlashAttention is an IO-aware exact attention algorithm. It reduces memory
traffic between high-bandwidth memory and on-chip SRAM by tiling attention
computation instead of materializing the full attention matrix.

## Why It Matters

The main bottleneck in standard attention is not only floating point compute.
For long sequences, memory movement dominates latency. FlashAttention improves
training and inference efficiency by reducing unnecessary HBM reads and writes.

