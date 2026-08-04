# Continuous autotrain cycle 2 results (2026-08-04)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2` |
| Cycle intent | `retry_measurement` (frozen replay of cycle 1) |
| Source | `eba6db3044076285581b80cfe5294a2ecbcee8a1` |
| Integration | `0da92101e31ee0906e54fe11b206a2e7a27fbe6f` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Status | `structural_similarity` |
| --- | --- | --- |
| c2-control | complete, ship gate rejected (fixture scale) | 0.0575 |
| c2-bounds | complete, ship gate rejected (fixture scale) | 0.0575 |

Primary metric delta: **0.0** (identical control/candidate value).

## Outcome

This is the automatic `retry_measurement` replay of cycle 1's frozen control/bounds arms, queued
after cycle 1's `repair_harness` action (npm-ci bootstrap, see
`docs/design/continuous-openui-local-20260804-c1-results.md`) was acknowledged. Both arms now run
end to end and produce a full `scoreboard.json` / `gates.json` -- the harness_failure from cycle 1
is confirmed resolved (replay-proven).

Ship gates still reject on expected fixture-scale grounds:

- `smoke:insufficient_n actual=3 need>=20`
- `smoke:meaningful_program_rate actual=0.0 need>=0.66`
- `smoke:structural_similarity actual=0.0575 need>=0.35`
- `smoke:component_type_recall actual=0.0 need>=0.35`
- `smoke:ast_beq_rate actual=0.0 need>=0.2`
- `smoke:canonical_beq_rate actual=0.0 need>=0.1`
- `smoke:reward_score actual=0.0 need>=0.3`

Primary metric (`smoke.structural_similarity`) is identical between control and candidate
(0.0575 vs 0.0575), so there is no measured win either.

## SDLC Phase A classification

Per `sdlc` autotrain-iteration-delivery, this cycle is **non-positive**: fixture
`insufficient_n` and a null primary-metric delta are both explicitly listed as not-positive
conditions. **No stacked PR opens for this cycle.** Local commit + docs only.

## Next-run priorities

1. **model:** test the distinct size-matched `component-plan` quality hypothesis next
   (confidence 0.90).
2. **evaluation:** keep the matched control as the size-matched baseline every cycle
   (confidence 0.70).
3. **model:** rotate thrash recommendation across the lever bank instead of bounds-only
   (confidence 0.65, monitor).
4. **infrastructure:** soft ship-gate fails on fixture n never stop the continuous loop
   (confidence 0.80, monitor).

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2/`
- Runs: `.../runs/c20260804-continuous-openui-local-8c0b60dd-c2-control/`,
  `.../runs/c20260804-continuous-openui-local-8c0b60dd-c2-bounds/`
- JSON twin: `continuous-openui-local-20260804-c2-results.json`
