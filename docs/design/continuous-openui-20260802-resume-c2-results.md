# Continuous autotrain resumed-session cycle 2 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

> Note: this campaign's `campaign_id` string
> (`continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c2`) collides with the earlier
> same-day [`continuous-openui-20260802-c2-results.md`](continuous-openui-20260802-c2-results.md)
> — the id is deterministic (`sha256(loop_id)[:8]` + cycle index, not time-based) and
> `outputs/autoresearch/` is gitignored, so this fresh scheduled container had no local loop state
> and restarted cycle numbering from 1. This doc covers a **distinct** campaign run (different
> `source_commit`); the original c2 doc is unaffected and still accurate for its own run.

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c2` |
| Source | `189ef062050878a3afdf0bfa41df39d043c18d27` |
| Device | CPU |
| Steps | 20 |
| Intent | `retry_measurement` (replay of this session's cycle-1 frozen pair) |

## Run matrix

| Arm | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c2-control | 3 | 1.0 | 0.0 | 0.0575 | 1419.78 | eval completed; ship gates fail (insufficient n + quality floors) |
| c2-bounds | 3 | 1.0 | 0.0 | 0.0575 | 1370.23 | eval completed; ship gates fail (insufficient n + quality floors) |

## What this confirms

This is the `retry_measurement` replay of this session's cycle-1 control/bounds pair, now that
`node_modules/@agentv/core` was restored via `npm ci`. Both arms produced a real scoreboard this
time — the cycle-1 harness_failure did not recur. Ship gates fail as **expected diagnostics** on
the 3-example smoke fixture (`insufficient_n`, need ≥ 20) plus quality-threshold misses
(`meaningful_program_rate`, `structural_similarity`, `component_type_recall`, `ast_beq_rate`,
`canonical_beq_rate`, `reward_score` all below floor). Per the continuous-mode contract this is a
**soft failure** (`fixture_insufficient_n_alone`), not a loop terminator.

`primary_metric` (`smoke.structural_similarity`) is identical between control and bounds
(`0.0575`), so the driver classified this cycle **non-positive** with `stack_action =
no_stack_layer_non_positive` — correct: no code/lever changed between the two arms, so there is
nothing to ship.

## Next-run priorities

1. **model:** the ranked matrix's next hypothesis is the size-matched `component-plan` lever
   (`c2-component-plan`) — the bounds/control pair is now exhausted and cannot be replayed again
   without a new preregistered hypothesis.
2. **evaluation:** keep the matched control as the size-matched baseline every cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c2/`
- JSON twin: `continuous-openui-20260802-resume-c2-results.json`
