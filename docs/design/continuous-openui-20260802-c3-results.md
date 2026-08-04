# Continuous autotrain cycle 3 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c3` |
| Source | `451785f59599c14bea42fefbd153d12a054b4f76` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | smoke n | parse_rate | meaningful_program_rate | structural_similarity | ast_beq_rate | canonical_beq_rate | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | 3 | — | — | — | — | — | **decode timeout** (measurement incomplete) |
| c3-canvas | 3 | 1.0 | 0.333 | 0.231 | 0.0 | 0.0 | eval completed; ship gates fail (insufficient n + quality) |

## What this confirms

This is the first cycle run after the AgentV harness repair documented in
[`continuous-openui-20260802-c2-results.md`](continuous-openui-20260802-c2-results.md). `c3-canvas`
completed end-to-end and produced a real AgentEvals scoreboard instead of the prior hard
`RuntimeError` — the driver logged `executable_unblock:candidate_completed_after_control_error`,
confirming the fix.

`c3-control` hit a different, unrelated issue: a typed model decode timeout under the 3-minute
wall cap (`decode_timeout_count=3`), leaving its measurement incomplete. Fixture-scale ship gates
correctly fail on both arms (`n=3` vs the required `>=20`); this is expected and not a promotion
signal.

Driver classification: `SDLC_PHASE_A NON_POSITIVE` — control measurement stayed incomplete and
both arms are `fixture_insufficient_n`, so no stack layer opens for this cycle even though one arm
independently confirms the harness repair. Treated as authoritative per repo policy (no stack for
non-positive cycles).

## Next-run priorities

1. **infrastructure:** retry the identical frozen `c3-control`/`c3-canvas` pair once to check
   whether the control decode timeout reproduces or was a one-run timing artifact (queued
   `retry_measurement`).
2. **model_build:** if the control timeout reproduces consistently, open a `repair_harness` lane
   for decode wall-budget headroom under the 3-minute cap instead of continuing to retry.
3. **harness (separate):** dedicated repair pass for the 64 pre-existing `tests/test_evals`
   failures on `main` flagged in cycle 2's doc.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c3/`
- Runs: `.../runs/c20260802-continuous-openui-202608-39ee9cf7-c3-control/`,
  `.../runs/c20260802-continuous-openui-202608-39ee9cf7-c3-canvas/`
- JSON twin: `continuous-openui-20260802-c3-results.json`
