# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c1` (follows `continuous-loop-20260730-c2`) |
| Source | `41d874c76b9ed68f4c6d375366ea4398b95a0429` |
| Device | CPU |
| Steps | 25 / batch 2 / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (explicit) |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c1-control | bounds off | 3 | 1.0 | 0.0 | 4650.74 | eval completed; ship gates fail (insufficient n + quality) |
| c20260801-c1-bounds | bounds **on** | 3 | 1.0 | 0.0 | 3035.50 | eval completed; ship gates fail (same) |

Primary delta (bounds − control) p50 latency: **-1615.24 ms** (bounds faster).

## Diagnostics

1. The c2 footgun (continuous `compile_commands` defaulting `eval_version` to a missing `v1` suite) is already fixed upstream: `default_eval_version()` now probes on-disk suites and correctly resolves `e938_role_safe_all_targets_v2`.
2. First attempt this cycle failed closed on both arms: `RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER`. This was a sandbox setup gap (this checkout had not run `npm ci`), not a repo bug. Repaired by running `npm ci` with `NODE_OPTIONS` unset (the ambient `NODE_OPTIONS='--import tsx' --max-old-space-size=8192` value breaks bare `node`/`npm` invocation in this sandbox) and replaying the identical spec.
3. `grammar_completion_bounds=True` shows a large smoke p50 latency drop (-1615.24 ms) at steps=25, but `meaningful_program_rate` is **0.0 on both arms**. Per the quality-aware Phase A classifier, a latency win requires held mpr ≥ ~1/3; this cycle's mpr=0.0 fails that floor, so the win is correctly rejected as **non-positive** (`latency_win_rejected_low_mpr`). A fast-but-meaningless decode is not evidence of a real win.
4. Ship gates fail on both arms for expected fixture reasons (`insufficient_n` n=3<20, `meaningful_program_rate` below gate) — diagnostic, not terminal.

## SDLC Phase A

- positive: **False**
- stack_layer: **False** (`no_stack_layer_non_positive`)
- action: local commits + docs only, no new stacked PR layer this cycle.

## Next-run priorities

1. **model_build/train_data:** 25 steps on `wf_smoke_v2` still produces zero meaningful programs on either arm; raise step budget or move to a richer `train_version` before re-attributing `grammar_completion_bounds` latency effects to quality.
2. **evaluation:** once mpr is nonzero and held between arms, re-run the bounds/canvas comparison to see whether the latency delta survives with a quality floor.
3. **infrastructure:** document the `npm ci` + `NODE_OPTIONS` unset requirement for AgentV-backed evals in fresh checkouts so future continuous cycles do not rediscover it from a failed cycle.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c1/`
- Runs: `.../runs/c20260801-c1-control/`, `.../runs/c20260801-c1-bounds/`
- JSON twin: `continuous-openui-20260801-c1-results.json`
