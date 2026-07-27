---
type: concept
status: active
tags: [recurrence, diagnostics, fixture]
created: 2026-07-23
updated: 2026-07-27
linear: SLM-282, SLM-421, SLM-317, SLM-431
design: docs/design/iter-slm282-recurrence-health-20260723.md, docs/design/iter-slm282-recurrence-health-powered-rerun-20260725.md
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

**SLM-282 (n=2, 2026-07-23): negative.** Only seed 0 passed. For seed 1 / R=4 /
example `b`, CE regressed from `17.487688` at depth 3 to `17.855324` at depth
4, so the fixed two-of-two-seed prerequisite failed even though the
token-weighted aggregate improved. All required telemetry was finite, but
several local directional gains also exceed one. This n=2 record is immutable
and remains the historical evidence — PR #853/#854/#855/#856 correctly cited
it (alongside SLM-317's inconclusive value gate) to gate-close LAR3-01..04 as
`not_authorized`.

**SLM-421 (n=20, powered rerun, 2026-07-25): `recursive_core_positive`.** The
fixed 2-seed rule could not distinguish a real contraction violation from
fixture-scale noise, so SLM-421 added an additive Wilson-interval `power_rule`
to the same unchanged harness/model/data recipe and reran on 20 fresh seeds
(2..21, disjoint from SLM-282's 0/1) with a preregistered `min_pass_rate=0.5`
locked before any of those seeds were observed. Observed: 18/20 passed
(rate 0.90, Wilson 95% CI [0.699, 0.972]) — the lower bound clears 0.5, so the
`recursive_core_positive` LAR3-reopening condition is now satisfied for this
`as_is` shared_recursive-tower fixture configuration. The seed-1 regression
that drove the SLM-282 negative reads as fixture-scale noise at n=2, not a
systematic property of the core.

**Net effect on LAR3**: only one of the two PR #853-#856 reopening conditions
is now met. LAR3 stays closed — the second condition (a passing SLM-317-style
valid-state repair advancement screen) is unresolved; see the SLM-431 blocker
below. No ship, checkpoint, or production-default claim follows from either
record; residual_delta remains a fixture-only counterfactual, never a
production default.

**SLM-431 (2026-07-27): harness landed; powered rerun BLOCKED by unmerged
branch-only model dependencies (stop rule).** The SLM-317 harness
(`src/slm_training/harnesses/experiments/slm317_repair_hybrid.py`),
its runner (`scripts/run_slm317_repair_hybrid.py`), its tests, and the frozen
`docs/design/iter-slm317-repair-hybrid-20260724.{md,json}` record were landed
byte-identical from unmerged branch commit `48e5cadc`; the 20 original tests
pass unmodified against main HEAD, and SLM-431 added the same additive,
default-off `--min-pass-rate` Wilson power-rule surface SLM-421 added to
`run_slm138_recursive_denoiser_fixture.py` (7 new deterministic helper
tests). However the powered rerun itself **cannot execute on main**: the
runner's improved/historical repair arms construct `TreeEditDiffusionConfig`
with `value_label_mode` and `stop_slot_accounting` knobs that exist **only**
in the branch's forked `tree_edit_diffusion.py` (unmerged SLM-305/308/310
branch commits; main's model has neither the knobs nor the concepts under
any name). Running the rerun would require porting those branch model
changes — a behavior change the issue explicitly forbids. Per the issue's
falsification/stop rule this is reported honestly and the rerun was **not**
forced: no `iter-slm317-repair-hybrid-powered-rerun-*` artifact exists, no
disposition (repair_positive / repair_negative / inconclusive_underpowered)
was produced, and the second LAR3-reopening condition remains **unmet**. Next
step: a decision issue on whether to land/port the SLM-305/308/310 model
work (or re-express the historical/improved arms against main's model) before
any powered rerun is attempted.
