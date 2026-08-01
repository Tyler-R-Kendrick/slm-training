# Continuous autotrain cycle 1 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260801-c1` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | 20 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | --- |
| c20260801-c1-control | bounds off | 1.0 | 0.0 | 2709.07 | eval completed; ship gates fail (fixture n + quality) |
| c20260801-c1-bounds | bounds **on** | 1.0 | 0.0 | 2756.66 | eval completed; ship gates fail (same) |

Primary delta (bounds − control) p50 latency: **+47.59 ms** (positive = bounds slower).

## Diagnostics

1. First cycle of a fresh continuous loop worktree run: the AgentV ship-gate
   publication path (`evaluate_model.py --ship-gates` → `publish_agentv_evaluation`)
   failed closed once with `AgentV SDK is unavailable; run npm ci in the checkout
   or set AGENTV_RUNNER`, because `node_modules/` had not been installed in this
   checkout yet. This is an environment-setup gap, not a harness or model bug.
2. Primary metrics for this cycle came from the campaign's own decode
   measurement path (unaffected by the AgentV publish failure), so the arms
   still produced a usable comparison.
3. `grammar_completion_bounds=True` did **not** improve smoke decode p50 under
   this recipe; `meaningful_program_rate` stayed at 0.0 for both arms at
   fixture scale (n below the 20-sample ship-gate threshold).
4. Fixed for cycle 2: ran `npm ci` with `NODE_OPTIONS=--max-old-space-size=8192`
   (the ambient `NODE_OPTIONS` in this container ships a malformed quoted
   `--import tsx` value that node's CLI parser rejects verbatim).

## Next-run priorities

1. **infrastructure:** keep `npm ci` in the loop worktree bootstrap so AgentV
   ship-gate publication does not fail closed on a fresh checkout (done for
   cycle 2 onward this session).
2. **model:** re-test `grammar_completion_bounds` only after a higher-step
   budget clears fixture `insufficient_n`.
3. **evaluation:** keep ship gates honest; do not weaken for continuous smoke.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c1/`
- JSON twin: `continuous-openui-20260801-c1-results.json`
