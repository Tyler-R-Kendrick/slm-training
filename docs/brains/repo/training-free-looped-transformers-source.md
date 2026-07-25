---
type: source
status: mapped
tags: [source, recurrence, transformers]
created: 2026-07-23
citation: Lizhang Chen, Jonathan Li, Chen Liang, Ni Lao, and Qiang Liu. Training-Free Looped Transformers, arXiv 2026.
url: https://arxiv.org/abs/2605.23872
fidelity: Adjacent
lineage: docs/design/research-lineage.md, Shared recursive denoiser tower
---

# Training-Free Looped Transformers

## One-line

Naively reapplying frozen Transformer blocks can degrade quality; the loop
application rule itself is an experimental variable.

## What we could take

Make a raw block reapplication control explicit and isolate the block's
residual update as a diagnostic counterfactual. SLM-282 therefore compares the
historical `as_is` recurrence with fixture-only `residual_delta`.

## What we would NOT take

The paper retrofits recurrence at inference time into frozen autoregressive
models and uses damped substeps. OpenUI trains a masked denoiser end to end.
`residual_delta = block(x) - x` is an inspired diagnostic, not the paper's
method and never a production default.

## Connections

Supports `[[recursive-recurrence-health]]`; complements
`[[deeploop-source]]`.

The mapped fixture result is summarized in the
[`quality-experiment-matrix.md`](../../design/quality-experiment-matrix.md)
SLM-282 section.
