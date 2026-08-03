# Continuous autotrain: 2026-08-03 (scheduled session ctyadc) cycle 3 — component-plan structural win, frozen-replay reproduction (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c3`
**Integration commit:** `3e042a27` (this session's cycle-2 docs commit, on top of `main` tip `45d78cb4`)
**Frozen replay of:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`

**Verdict:** `component-plan` beats its size-matched control on the declared
primary at this seed — the same primary-metric delta, at the same seed, as
every prior independently-run measurement of the identical hypothesis.
Fixture screening only — not a ship or promotion claim.

| Arm | Seed | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | 0 | 30965.21 |
| component-plan | 100002 | .38280 | 0 | 23572.64 |

Primary improvement `+.05613` (`0.32666666666666666 -> 0.38280000000000003`).

## Closes out cycle 2's incomplete measurement

This cycle is the driver's own `retry_measurement` action, consumed
automatically after `git fetch`/merge: it reused cycle 2's `component-plan`
checkpoint (`20e573b1...f0f8a8e741`, no retrain — see
`reused_checkpoint_from` in the JSON) and freshly trained a matching control
(`6abf57d4...db3512b`), completing the attribution that this session's own
mid-cycle `git commit --amend` interrupted in
[`continuous-openui-local-ctyadc-c2-results.md`](continuous-openui-local-ctyadc-c2-results.md).
No further git history mutation occurred during this cycle.

## Yet another reproduction of the same hypothesis

Byte-identical to every prior independent run of this hypothesis:

1. [`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
   (merged as PR #1369).
2. [`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md).
3. [`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)
   (merged as part of PR #1376).
4. [`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md)
   (merged as part of PR #1378).
5. [`continuous-openui-local-sk4t9p-c2-results.md`](continuous-openui-local-sk4t9p-c2-results.md)
   (session `sk4t9p`, branch `claude/great-dirac-j34ebs` — not yet merged into
   `main` as of this session; referenced for provenance, not assumed landed).

Checkpoint SHAs are byte-identical across every one of these sessions,
confirming the underlying fixture effect is deterministic and reproducible
across intervening harness commits. `meaningful_program_rate` and
`binder_reference_f1` both stay `0` on both arms — the win stays confined to
raw structural similarity, not full program correctness.

Ship gates fail as expected: `insufficient_n` (n=3, need 20).

## Driver classifier note

The automated `sdlc_delivery.json` classifier flagged
`positive_no_tracked_delta_skip_stack` because this cycle re-ran the
identical frozen recipe with no new code/harness diff of its own. Per `sdlc`
autotrain-iteration-delivery, documenting a positive result is itself the
reviewable delta that earns a PR.

## SDLC Phase A

**Positive** (`primary_metric_win`). Documenting this result creates the
reviewable delta required to open a PR for this session's cycles (bundled
with this branch's cycle 1 null-delta doc and cycle 2 measurement-incomplete
doc).

## Next priorities

1. This hypothesis is now exhausted for screening; test the driver's ranked
   `component-edge` hypothesis next (rank 1, confidence 0.9) rather than
   re-running `component-plan` again.
2. Keep the matched control fixed every cycle (rank 2, confidence 0.7).

Machine evidence:
[`continuous-openui-local-ctyadc-c3-results.json`](continuous-openui-local-ctyadc-c3-results.json).
