---
title: "Speculative Decoding Notes"
source_type: "markdown"
topic: "inference"
difficulty: "intermediate"
tags: ["decoding", "serving", "latency"]
workspace_id: "public"
visibility: "public"
---

# Speculative Decoding

Speculative decoding accelerates autoregressive inference by using a small
draft model to propose several candidate tokens and a larger target model to
verify them in parallel. The method can reduce latency while preserving the
target model distribution when verification accepts or rejects draft tokens
correctly.

## Speedup Factors

Speculative decoding speedup depends on draft model quality, acceptance rate,
and verification overhead. A better draft model proposes more tokens that the
target model accepts, but the draft model must still be cheap enough that its
extra work does not erase the latency gain.
