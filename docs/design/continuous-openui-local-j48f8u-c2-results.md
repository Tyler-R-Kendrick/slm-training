# Continuous autotrain: 2026-08-03 (session j48f8u) cycle 2 — component-plan structural win, third reproduction (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `d367ddfe` (this session's cycle-1 docs commit, on top
of `main` tip `318492c5`)

**Verdict:** `component-plan` beats its size-matched control on the declared
primary at this seed — the same primary-metric delta, at the same seed, as
two prior independently-run sessions' measurements of the identical
hypothesis. Fixture screening only — not a ship or promotion claim, and
**not** a confirmation of the still-blocked champion candidate (see below).

| Arm | Seed | structural_similarity | component_type_recall | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | .16667 | 14554.69 |
| component-plan | 100002 | .38280 | .16667 | 12549.58 |

Primary improvement `+.05613` (`0.32666666666666666 -> 0.38280000000000003`)
— byte-identical to two prior sessions' independent runs of this same
hypothesis:

1. [`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
   (merged as PR #1369, "c1-c2 component-plan structural win (+0.056)").
2. [`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md).

The delta reproducing exactly across three independent sessions, including
this one running on top of several intervening harness commits, is strong
evidence the underlying fixture effect is real and deterministic rather than
sampling noise. `meaningful_program_rate`, `binder_reference_f1`,
`placeholder_fidelity`, `ast_beq_rate`, `canonical_beq_rate`, and
`reward_score` are **0 on both arms** — the win stays confined to raw
structural similarity, not full program correctness.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` were not run.

## This is not a confirmation attempt

A prior session already tried to fresh-seed-confirm this exact hypothesis
(the `c4` champion candidate, `champ-continuous-openui-local-4-2694d77fc99953e4`)
and hit two distinct harness blockers, documented in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md):

1. A dual-arm decode timeout at seed `100005` with an undetermined root
   cause.
2. `_apply_frozen_replay` does not recognize `-confirm`-suffixed arm slugs,
   so the driver's own recommended remedy for (1) is currently inexecutable.

That doc explicitly asks for a **dedicated `improve-openui-harnesses`
session** with room to investigate, and says not to attempt further
speculative fixes without it. This cycle does **not** attempt to fix either
blocker — it only adds a third independent screening-stage reproduction of
the same primary-metric win under a fresh campaign id.

## SDLC Phase A

**Positive** (`primary_metric_win`). Per `sdlc` autotrain-iteration-delivery,
documenting this result creates the reviewable delta required to open a
stacked layer for this cycle.

## Next priorities

1. Do **not** attempt the c5/c6 frozen-replay `-confirm` slug bug or the
   seed-`100005` dual-arm decode timeout speculatively; route to
   `improve-openui-harnesses` with dedicated investigation time.
2. Once those blockers are resolved: confirm the fixture candidate on a
   fresh seed with the exact size-matched recipe before any promotion.
3. Keep promotion formal preflight locked until fresh confirmation
   establishes a champion.

Machine evidence:
[`continuous-openui-local-j48f8u-c2-results.json`](continuous-openui-local-j48f8u-c2-results.json).
