# Continuous autotrain cycle 2 results (2026-08-02, loop pq269u)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-pq269u` |
| Campaign | `continuous-loop-20260802-continuous-openui-pq269u-2619fa49-c2` |
| Source | `9ba7f7baba1e3858e0ef484a86093dcb1d351ae9` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Status |
| --- | --- |
| c2-control | trained; eval ran end-to-end, **measurement_incomplete** (n=3/32, all decode timeouts) |
| c2-bounds | trained; eval ran end-to-end, **measurement_incomplete** (n=3/32, all decode timeouts) |

## Result: c1 harness repair confirmed

This cycle replays the same recipe that hit `harness_failure` in c1. The AgentV publish step now
runs to completion and writes a real `scoreboard.json` / `gates.json` with 10 AgentEvals criteria
evaluated, instead of raising before any measurement — a **proven executable unblock** for the
`npm ci` + `NODE_OPTIONS` sanitization repair landed this cycle chain.

## New (unrelated) constraint: decode timeout under the wall cap

Both arms only completed 3 of the requested 32 smoke documents before the 3-minute wall cap, and
all 3 timed out during decode (`decode_timeout_count=3`). With zero real generations scored, every
quality metric (`meaningful_program_rate`, `structural_similarity`, `ast_beq_rate`, …) is `null`
and the ship gates fail closed on `insufficient_n`. This is the fixture-scale
decode-timeout-vs-wall-cap tension already tracked by earlier continuous-loop threads, not a new
defect — flagged here as a soft failure per the loop rules ("ship gates fail on fixture n" never
stops the loop).

## Next-run priorities

1. **infrastructure:** investigate whether a lower `--decode-timeout-seconds` or a smaller
   requested smoke `n` lets more documents complete within the 3-minute wall cap, or whether this
   recipe needs a longer wall budget than `MAX_RUN_MINUTES` allows for smoke-scale evaluation.
2. **process:** route any canonical eval-runtime change through `improve-openui-harnesses`.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-pq269u-2619fa49-c2/`
- Runs: `.../runs/c20260802-continuous-openui-pq269u-2619fa49-c2-control/`,
  `.../runs/c20260802-continuous-openui-pq269u-2619fa49-c2-bounds/`
- JSON twin: `continuous-openui-pq269u-c2-results.json`
