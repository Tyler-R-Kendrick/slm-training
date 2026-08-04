# Continuous autotrain cycle 1 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c1` |
| Source | `c1c4eca349b66f05684975575a3640ced50051ea` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | latency_ms_p50 | parse_rate | Status |
| --- | --- | ---: | ---: | --- |
| c1-control | bounds off | 2784.59 | 1.0 | **failed** — AgentV SDK unavailable (infrastructure) |
| c1-bounds | bounds on | 2873.60 | 1.0 | **failed** — same infrastructure blocker |

Primary delta (bounds − control) p50 latency: **89.01 ms** — not meaningful; neither
arm produced a ship-gate scoreboard.

## Diagnostics

1. Fresh container had no `node_modules` for the AgentEvals/`@agentv/core` eval
   runner: `RuntimeError: AgentV SDK is unavailable; run npm ci in the
   checkout or set AGENTV_RUNNER` (`src/slm_training/evals/agentv.py`).
2. The ambient `NODE_OPTIONS` env var (`--import tsx" --max-old-space-size=8192`)
   is malformed and breaks bare `node`/`npm` invocations independent of the
   AgentV gap.
3. Self-healed in-session: ran `npm ci` with `NODE_OPTIONS` unset for the
   subprocess. No repository code change was required for this session — an
   open PR (#1254, "sanitize AgentV NODE_OPTIONS") already carries the
   in-harness fix pending merge.
4. Re-ran the identical `train_version`/`eval_version` recipe as cycle c2
   (below) after the fix; evals published successfully there.

## Next-run priorities

1. **infrastructure:** land #1254 so continuous-loop containers don't need a
   manual `npm ci` self-heal.
2. **model:** do not use this cycle's pre-crash latency numbers as model
   evidence — treat cycle c2 as the first clean read for this loop.
3. **process:** empty-metrics / infrastructure-diagnosed arms stay
   `harness_failure`, never `rejected`.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c1/`
- JSON twin: `continuous-openui-20260801-c1-results.json`
