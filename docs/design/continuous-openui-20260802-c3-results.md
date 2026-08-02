# Continuous autotrain cycle 3 results (2026-08-02, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3` |
| Source | `2500b717855d32ee82836d9807d0a7a2a570951b` |
| Cycle intent | `retry_measurement` (frozen replay of cycle 1's checkpoints, post-fix) |
| Train | `wf_smoke_v2`, 21 steps / batch 2 / seed 100001 (reused checkpoints) |
| Eval | `e938_role_safe_all_targets_v2` |

This is the first cycle in the `continuous-openui-local` lineage to produce a
**complete** AgentV-backed measurement — the NODE_OPTIONS fix from cycle 2
held.

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 1537.27 | reject (insufficient n, quality) |
| c3-bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.0575 | 1624.73 | reject (same) |

Primary metric delta (bounds − control) on `smoke.structural_similarity`:
**0.0** (exact tie). Latency delta: bounds is **5.68% slower** (1624.73 vs
1537.27 ms).

## Diagnostics

1. Both arms reused cycle 1's trained checkpoints (`FROZEN_REPLAY_ACK`); only
   evaluation re-ran, now with `NODE_OPTIONS` stripped from the AgentV
   subprocess env.
2. Ship gates reject both arms as expected: fixture `n=3` is below the
   policy's `min_n=20` floor, and quality metrics (`meaningful_program_rate`,
   `structural_similarity`, `ast_beq_rate`, `canonical_beq_rate`,
   `reward_score`) are all below threshold on this 21-step scratch model.
   This is a screening-scale artifact, not a promotion candidate.
3. `grammar_completion_bounds=True` produced **no quality change** and a
   **latency regression** on this recipe — rejected as a candidate lever
   here.

## Next-run priorities

1. **model:** test the size-matched `component-plan` supervision hypothesis
   next (driver's top-ranked priority).
2. **evaluation:** keep the matched control as the baseline every cycle to
   avoid recipe drift.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the
   loop; continue.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c3/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c3-{control,bounds}/`
- JSON twin: `continuous-openui-20260802-c3-results.json`
