---
title: "PagedAttention Notes"
source_type: "markdown"
topic: "inference"
difficulty: "intermediate"
tags: ["kv-cache", "serving", "memory"]
workspace_id: "public"
visibility: "public"
---

# PagedAttention

PagedAttention is a memory management technique for LLM serving. It stores the
KV cache in fixed-size blocks, similar to virtual memory paging, so requests can
grow without requiring large contiguous memory allocations.

## KV Cache Fragmentation

During autoregressive decoding, each sequence grows token by token. Without a
paged layout, KV-cache memory can become fragmented or overallocated. Paging
improves utilization and makes continuous batching easier to manage.

