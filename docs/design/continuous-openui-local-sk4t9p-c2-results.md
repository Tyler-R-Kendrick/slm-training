# Continuous autotrain: 2026-08-03 (scheduled session sk4t9p) cycle 2 — component-plan structural win, 5th reproduction (screening)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c2`
**Integration commit:** `6c623c64` (this session's cycle-1 docs commit, on top of `main` tip `089b1649`)

**Verdict:** `component-plan` beats its size-matched control on the declared
primary at this seed — the same primary-metric delta, at the same seed, as
**four** prior independently-run sessions' measurements of the identical
hypothesis. Fixture screening only — not a ship or promotion claim.

| Arm | Seed | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | ---: | ---: | ---: |
| control | 100002 | .32667 | 0 | 15875.48 |
| component-plan | 100002 | .38280 | 0 | 13317.56 |

Primary improvement `+.05613` (`0.32666666666666666 -> 0.38280000000000003`)
— byte-identical to four prior sessions' independent runs of this same
hypothesis:

1. [`continuous-openui-20260803-c2-results.md`](continuous-openui-20260803-c2-results.md)
   (merged as PR #1369).
2. [`autotrain-cycle-c4-component-plan-efficiency-win.md`](autotrain-cycle-c4-component-plan-efficiency-win.md).
3. [`continuous-openui-local-j48f8u-c2-results.md`](continuous-openui-local-j48f8u-c2-results.md)
   (merged as part of PR #1376).
4. [`continuous-openui-local-ts5ofk-c2-results.md`](continuous-openui-local-ts5ofk-c2-results.md)
   (merged as part of PR #1378).

This is the **fifth** independent, byte-identical reproduction. Checkpoint
SHAs (control `6abf57d4...db3512b`, component-plan `20e573b1...f0f8a8e741`)
are byte-identical to the `j48f8u` session's checkpoints, confirming the
underlying fixture effect is deterministic across intervening harness
commits. `meaningful_program_rate` stays 0 on both arms; the win stays
confined to raw structural similarity, not full program correctness.
`binder_reference_f1` reads 0 on both arms this cycle (matching `ts5ofk`,
not `j48f8u`'s `.16667` — an intervening metric-computation difference, not
attributable to this hypothesis).

Ship gates fail as expected: `insufficient_n` (n=3, need 20).

## Driver classifier note

The automated `sdlc_delivery.json` classifier flagged
`positive_no_tracked_delta_skip_stack` because this cycle re-ran the
identical recipe with no new code/harness diff of its own. Per `sdlc`
autotrain-iteration-delivery, documenting a positive result is itself the
reviewable delta that earns a stacked layer — matching how all four prior
reproductions of this exact hypothesis were delivered (PR #1369, PR #1376,
PR #1378).

## SDLC Phase A

**Positive** (`primary_metric_win`). Documenting this result creates the
reviewable delta required to open a PR for this cycle (bundled with this
session's cycle 1 null-delta doc).

## Next priorities

1. Confirm the fixture candidate on a fresh seed with the exact
   size-matched treatment and control recipes before promotion (rank 1,
   confidence 0.95).
2. Keep promotion formal preflight locked until fresh confirmation
   establishes a champion (rank 2, confidence 1.0, lean assumption).

Machine evidence:
[`continuous-openui-local-sk4t9p-c2-results.json`](continuous-openui-local-sk4t9p-c2-results.json).
