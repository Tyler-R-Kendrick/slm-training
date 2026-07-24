---
type: concept
status: dead-end
tags: [data-augmentation, root-structure, lexer, negative-result]
created: 2026-07-24
linear:
design: ../../design/iter-e1088-root-tail-pair-20260724.md
sources: []
---

# Dependency-closed root-tail-pair augmentation

## Claim

Projecting the final two root siblings of generated `Stack` programs, with their
transitive binder closure, would improve compact root-list generalization without
using prompt markers or held-out examples.

## Why it might be true

The transform is grammar-derived, preserves dependency closure, and its strict
builder decontaminates structurally overlapping candidates. It was intended to
add non-leaking evidence for compact root composition.

## Falsification boundary

A fresh matched scratch train must not regress the pre-specified held structural
diagnostic under the same compiler policy. The measured outcome is recorded in
the linked design evidence.

## Status & next step

Dead end: do not repeat the root-tail projection, weight sweeps, or a decoder
overlay as evidence for it. Choose a distinct upstream structural mechanism;
consult the measured design record rather than reproducing its metrics here.
