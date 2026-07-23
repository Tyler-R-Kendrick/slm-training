---
type: concept
status: dead-end
tags: [recurrence, diagnostics, fixture]
created: 2026-07-23
linear: SLM-282
design: docs/design/iter-slm282-recurrence-health-20260723.md
sources: "[[deeploop-source]], [[training-free-looped-transformers-source]], https://arxiv.org/abs/2106.14342"
---

# Recursive recurrence health

## Claim

The canonical shared-recursive core should have finite per-depth state/update
telemetry and non-increasing masked CE through every anytime depth on at least
two seeded fixture runs before deeper recurrence experiments are activated.

## Why it might be true

Tied block visits create stability behavior that ordinary depth accounting can
miss (`[[deeploop-source]]`), while naive reapplication can regress
(`[[training-free-looped-transformers-source]]`). Equilibrium-model work also
connects fixed-point stability to update Jacobians, but SLM-282 deliberately
uses only a seeded local directional finite-difference proxy.

## Falsification boundary

Fail if either seeded `as_is` run violates
`CE(final) <= CE(previous) <= CE(r=1)`, if any required y/z update ratio is
non-finite, or if matched initialization/data/optimizer controls diverge.

## Status & next step

Negative. Only seed 0 passed. For seed 1 / R=4 / example `b`, CE regressed from
`17.487688` at depth 3 to `17.855324` at depth 4, so the two-seed prerequisite
failed even though the token-weighted aggregate improved. All required
telemetry was finite, but several local directional gains also exceed one.
Do not activate LAR3 from this result; no ship, checkpoint, or
production-default claim follows.
