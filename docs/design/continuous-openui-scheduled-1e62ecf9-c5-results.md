# Continuous autotrain: 2026-08-04 (scheduled loop `1e62ecf9`) cycle 5 — component-edge quality-neutral

**Loop:** `continuous-openui-scheduled`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-1e62ecf9-c5`
**Integration commit:** `a9a1c52` (cycle 4's docs commit)
**Objective:** distinct size-matched `component-edge` quality hypothesis (rank-1
priority carried from [cycle 4](continuous-openui-scheduled-1e62ecf9-c4-results.md))

## Result

| Arm | structural_similarity | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | trainable_params |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.043067 | 0.0 | 0.822222 | 16174.65 | 1,766,987 |
| component-edge (candidate) | 0.043067 | 0.0 | 0.822222 | 14592.06 | 1,766,987 |

Checkpoint SHA256: control `0e3d1953c500155696814acc0bcf72fb5285ad39b076e9be3653167008e64ca2`,
component-edge `aebbafec07f56353fdb99448b1f5e6f4ccef6594bc54a74b3b96a64d1d8416c0`
(both local under `outputs/autoresearch/.../runs/`, explicit no-sync).

Guarded quality is an exact tie between control and candidate (both arms
also score `meaningful_program_rate=0`, well below this loop's typical
`.333`, on this particular seed's fixture sample). The candidate is ~9.8%
faster (p50), but with zero quality signal a latency-only delta is not a
metric win per the SDLC tradeoff rule. Fixture scale (`n=3`, need >=20)
still fails the honest ship-gate volume floor as expected — not a ship
claim.

## SDLC Phase A

**Non-positive**: primary metric (`smoke.structural_similarity`) is an
exact null tie (`0.043067` both), `fixture_insufficient_n` on both arms.
This is the third consecutive null screening result in this loop
(c3 `both`, c4 `component-plan`, c5 `component-edge`) — a routine outcome at
this fixture scale, not a repeated hard block (each cycle completed cleanly
with a distinct, honestly measured knob bank; no harness error recurred).
Local commit + docs only, no stack layer.

## Next priority (rank 1, confidence 0.90)

The driver's rotation proposes re-testing `component-plan`
(`c20260804-continuous-openui-schedu-1e62ecf9-c5-component-plan`) with a
fresh seed next, per its own thrash-bank rotation rule.
