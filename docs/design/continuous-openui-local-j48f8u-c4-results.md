# Continuous autotrain: 2026-08-03 (session j48f8u) cycle 4 — batch-size exact tie, non-positive

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c4`
**Integration commit:** `0b57e1e8` (this session's cycle-3 docs commit, on
top of `main` tip `d5b0c318`)

**Verdict:** `batch1` (`batch_size=1`) ties its size-matched control
(`batch_size=2`) exactly on every smoke quality metric despite a materially
different training loss trajectory. Fixture screening only — not a ship or
promotion claim.

| Arm | batch_size | last_loss | structural_similarity | binder_reference_f1 | reward_score | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 2 | 8.852 | .41667 | .95238 | .936 | 11142.66 |
| batch1 | 1 | 13.509 | .41667 | .95238 | .936 | 11538.82 |

Both arms parse all 3 documents and produce byte-identical quality metrics
(`structural_similarity`, `meaningful_program_rate`, `component_type_recall`,
`binder_reference_f1`, `placeholder_fidelity`, `reward_score` all tie to 5
decimal places) despite `batch1`'s training loss ending materially higher
(13.509 vs 8.852). `batch1` is also 396ms slower p50. Training loss diverges
between arms but eval-time program quality does not — consistent with this
session's [c3 finding](continuous-openui-local-j48f8u-c3-results.md) that
training loss is not a reliable promotion proxy for certified structural
quality.

Ship gates fail as expected: `insufficient_n` (n=3, need 20), and
`held_out`/`adversarial`/`ood`/`rico_held` were not run.

## SDLC Phase A

**Non-positive** (null primary-metric delta,
`smoke.structural_similarity` control=candidate=`.41667`). No stacked PR
layer for this cycle — local commit and docs only.

## Next priorities

1. Test the distinct size-matched `component-edge` quality hypothesis next
   rather than re-running the now-exhausted batch-size knob.
2. Keep the matched control as the size-matched baseline every cycle.
3. Rotate thrash recommendation across the lever bank rather than a single
   knob family.

Machine evidence:
[`continuous-openui-local-j48f8u-c4-results.json`](continuous-openui-local-j48f8u-c4-results.json).
