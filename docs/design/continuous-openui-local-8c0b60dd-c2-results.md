# Continuous autotrain: 2026-08-05 (session 8c0b60dd, scheduled run) cycle 2 — component-plan structural win, 7th+ reproduction (screening, candidate queued)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `9614a54` (cycle 1's docs commit, `origin/main` tip unchanged)

**Verdict:** the size-matched `component-plan` arm beats its control on the
declared primary at this seed — **positive**, but no tracked code/docs delta
beyond this record, so no stack layer opens. Candidate enqueued; promotion
stays formally locked pending fresh-seed confirmation.

| Arm | Seed | structural_similarity | parse_rate | component_type_recall | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | 1.0 | .16667 | 35873.13 |
| component-plan | 100002 | .38280 | 1.0 | .16667 | 32179.22 |

Primary delta `+0.0561`. Ship gates still fail as expected:
`fixture_insufficient_n` (n=3, need 20) plus the missing
`held_out`/`adversarial`/`ood`/`rico_held` suites — not ship evidence.

## Same win, at least the seventh reproduction

This is the same structural-quality win already reproduced across many prior
sessions (`ce27597` 5th reproduction, `4549cd8` 6th reproduction, and
predecessors) — and rejected every time it reached fresh-seed confirmation,
most recently in
[`6d97009` (session peuum8)](continuous-openui-local-peuum8-c1-results.md).
The candidate is queued again (`champ-continuous-openui-local-2-6cfba5d6fd08579f`)
but the Lean promotion preflight stays locked (confidence 1.00) until a fresh
seed independently reproduces it.

## SDLC Phase A

**Positive** (`primary_metric_win`), but `stack_layer=False`
(`positive_no_tracked_delta_skip_stack`): the win came from an existing knob
combination, not a code change, so there is nothing to stack a PR for. Docs
land locally and the loop continues into cycle 3.

## Next priorities

1. Confirm the fixture candidate on a fresh seed with the exact size-matched
   treatment/control recipes before promotion (rank 1, confidence 0.95).
2. Keep promotion formal preflight locked until fresh confirmation
   establishes a champion (rank 2, confidence 1.0, `lean_assumption`).

Machine evidence:
[`continuous-openui-local-8c0b60dd-c2-results.json`](continuous-openui-local-8c0b60dd-c2-results.json).
