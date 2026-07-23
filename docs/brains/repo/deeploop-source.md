---
type: source
status: mapped
tags: [source, recurrence, transformers]
created: 2026-07-23
citation: Shuzhen Li, Yifan Zhang, Jiacheng Guo, Quanquan Gu, and Mengdi Wang. DeepLoop, arXiv 2026.
url: https://arxiv.org/abs/2607.13491
fidelity: Adjacent
lineage: docs/design/research-lineage.md, Shared recursive denoiser tower
---

# DeepLoop: Depth Scaling for Looped Transformers

## One-line

Repeated visits to tied Transformer blocks change residual-scaling behavior, so
nominal unrolled depth alone is not enough to reason about recurrent stability.

## What we could take

Treat loop visits as a distinct stability axis and inspect every visited depth.
That motivated SLM-282's raw anytime-depth CE and state/update telemetry.

## What we would NOT take

DeepLoop studies a Post-LN GPT-style architecture and proposes a specific
visit-aware scaling rule. It does not establish contraction for OpenUI's
RMSNorm coupled y/z denoiser, and SLM-282 neither implements nor reproduces its
scaling recipe.

## Connections

Supports `[[recursive-recurrence-health]]`; complements
`[[training-free-looped-transformers-source]]`.

The mapped fixture result is summarized in the
[`quality-experiment-matrix.md`](../../design/quality-experiment-matrix.md)
SLM-282 section.
