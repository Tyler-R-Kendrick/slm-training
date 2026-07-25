---
type: concept
status: supported
tags: [recursive-denoiser, deep-supervision, objective-semantics]
created: 2026-07-23
linear: SLM-279
design: docs/design/iter-slm279-depth-supervision-correction-20260723.md
sources:
  - https://arxiv.org/html/2606.18022#S3.SS2
  - https://proceedings.mlr.press/v38/lee15a.html
---

# Recursive depth-supervision objective

## Claim

New OpenUI recursive-denoiser runs use final-depth reconstruction as the
primary term and may add normalized supervision over only depths `0..R-2`
under an explicit coefficient (`recursive_depth_aux_mode="intermediate_only"`).
Counting the final depth again is a deliberate `all_depths` experiment, never
an implicit default.

## Why it might be true

Recursive masked diffusion exposes multiple loop outputs and studies final,
all-loop, weighted, and truncated supervision as separate objectives. Deeply
supervised networks likewise motivate auxiliary losses at intermediate
representations. Those sources support making loop-loss semantics explicit;
they do not establish that one mode wins for OpenUI. The repo therefore uses
the least ambiguous canonical contract while retaining `all_depths` as an
experiment arm and `legacy_all_depths` solely for old-checkpoint reproduction.

The SLM-279 fixture proves the arithmetic and decomposition, including zero
final-depth auxiliary contribution. It is correction-only evidence, not a
quality result.

## Falsification boundary

The contract is broken if the exact reducer matrix fails, a zero-weighted depth
receives gradient, `intermediate_only` reads `depth_logits[-1]`, the telemetry
sum identities fail, or a pre-field checkpoint no longer migrates to the old
all-depth semantics. A future full-suite experiment may reject the canonical
choice for quality, but only with honest multi-suite evidence; that would not
invalidate the arithmetic contract.

## Status & next step

Supported as a correctness and compatibility contract by SLM-279. The next
quality question is a bounded, explicitly labeled `intermediate_only` versus
`all_depths` comparison on real held-out suites; no such winner is claimed by
the fixture.
