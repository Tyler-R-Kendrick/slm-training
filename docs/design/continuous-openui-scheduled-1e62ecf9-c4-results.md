# Continuous autotrain: 2026-08-04 (scheduled loop `1e62ecf9`) cycle 4 — component-plan quality-neutral

**Loop:** `continuous-openui-scheduled`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-1e62ecf9-c4`
**Integration commit:** `0f4f552` (cycle 3's docs commit)
**Objective:** distinct size-matched `component-plan` quality hypothesis (rank-1
priority carried from [cycle 3](continuous-openui-scheduled-1e62ecf9-c3-results.md))

## Result

| Arm | structural_similarity | meaningful_program_rate | binder_reference_f1 | latency_ms_p50 | trainable_params |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 0.416667 | 0.333333 | 0.952381 | 22500.28 | 1,755,764 |
| component-plan (candidate) | 0.416667 | 0.333333 | 0.952381 | 26692.86 | 1,755,764 |

Checkpoint SHA256: control `7cad65d11dffc1dfe7a41501337030c1bb52e3c0abac381adc1382ed42f5a989`,
component-plan `25b81d11520c5e4c2c9497bf791df9c7e2b3e444263cf65deee4fb2861c8f27c`
(both local under `outputs/autoresearch/.../runs/`, explicit no-sync).

Every guarded quality metric ties exactly between control and candidate; the
`component-plan` head adds no measurable structural/semantic signal on this
size-matched fixture arm and the candidate is ~18.6% slower (p50). Fixture
scale (`n=3`, need >=20) still fails the honest ship-gate volume floor as
expected — not a ship claim.

## SDLC Phase A

**Non-positive**: primary metric (`smoke.structural_similarity`) is an exact
null tie (`0.416667` both), and `fixture_insufficient_n` fires on both arms.
Per `sdlc` autotrain-iteration-delivery, a latency regression with no quality
signal is a straightforward reject — no stack layer. Local commit + docs only.

## Next priority (rank 1, confidence 0.90)

The completed non-positive arm is exhausted; the next cycle should test the
distinct size-matched `component-edge` quality hypothesis
(`c20260804-continuous-openui-schedu-1e62ecf9-c4-component-edge` proposed)
rather than repeat `component-plan` or `both`.
