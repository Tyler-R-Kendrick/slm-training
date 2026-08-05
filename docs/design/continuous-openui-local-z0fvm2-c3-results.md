# Continuous autotrain: 2026-08-05 (scheduled loop `z0fvm2`) cycle 3 — component-plan structural-similarity win reproduced again

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `b3693061` (this session's c2 docs commit on top of `bdf143cd`)
**Predecessor:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`

**Verdict:** SDLC Phase A **positive** — genuine primary-metric win, frozen
replay of cycle 2's arm pair after the control decode-timeout self-healed
(both arms completed cleanly this time; the incomplete-measurement state did
not reproduce).

## Results

| Arm | parse_rate | meaningful_program_rate | structural_similarity | latency p50 (ms) | ship gates |
| --- | --- | --- | --- | --- | --- |
| control | 1.0 | 0.0 | 0.32667 | 35231.83 | fail (gate reject) |
| component-plan | 1.0 | 0.0 | **0.38280** | 25520.60 | fail (gate reject) |

Primary metric (`smoke.structural_similarity`) improvement: **+0.05613**
(0.32667 → 0.38280). Candidate also faster p50 (25520.6ms vs 35231.83ms).
`component-plan` candidate sets `component_plan_decode_weight=1.0` and
`component_plan_loss_weight=1.0` (structural aux head profile
`component-plan`) vs. an all-zero-weight control at matched
`trainable_params=1755764`.

## SDLC Phase A

**Positive** (`primary_metric_win:smoke.structural_similarity:+0.05613`).
Champion enqueued: `champ-continuous-openui-local-3-6cfba5d6fd08579f`
(fingerprint `6cfba5d6fd08579f`), `climb_state=candidate_queued`,
`confirm_attempts=0` — **not** yet `climb_accepted`. This documentation
commit is the tracked delta; stacking a PR for this cycle per
`sdlc` autotrain-iteration-delivery.

## Important caveat — not a novel finding

This is the **same `component-plan` structural-aux-head candidate** already
documented across many prior sessions of this shared `continuous-openui-local`
loop, e.g.:

- `ce27597` "component-plan structural win, 5th reproduction"
- `4549cd8` "component-plan structural win, 6th reproduction"

...but the identical candidate has **also been rejected on fresh-seed
confirmation** in other sessions:

- `6d97009` "component-plan win rejected on fresh-seed confirmation"
- `528311e` "non-positive; falsifies queued champion"
- `7b2f64c` "component-plan rejected on fresh seed"

This session's win (`arm_seed=100002`) is one more screening-scale (`n=3`)
data point in that ongoing back-and-forth, not a confirmed promotion.
`meaningful_program_rate=0.0` on both arms — the win is on
`structural_similarity` + latency only. Honest ship gates fail as expected
(evidence volume: need `n>=20`).

## Next priorities

1. (rank 1) Test the distinct size-matched `component-edge` quality
   hypothesis next (`c20260805-continuous-openui-local-8c0b60dd-c3-component-edge`).
2. (rank 2) Keep the matched control as the size-matched baseline every cycle.

## Honesty

Fixture (`n=3`) screening evidence only. Not a ship claim. Not yet a
confirmed champion — multi-seed confirmation is still required before any
promotion.
