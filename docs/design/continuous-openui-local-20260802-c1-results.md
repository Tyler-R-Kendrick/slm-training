# Continuous autotrain cycle 1 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1` |
| Source | `b8188a49672f187146bf1a76a353cbe188f9b99d` |
| Device | CPU |
| Steps | 20 (recorded 21) / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2`, suite `smoke` |
| Wall cap | 3 minutes |
| Hypothesis | `grammar_completion_bounds` reduces smoke `latency_ms_p50` vs matched control without lowering `parse_rate` |
| Primary metric | `smoke.structural_similarity` (direction: increase, minimum_effect 0.01) |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c1-control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 1432.96 | eval **incomplete**: ship-gate `scoreboard.json` never written |
| c1-bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.0575 | 1357.41 | eval **incomplete**: ship-gate `scoreboard.json` never written |

Primary delta (bounds − control) `structural_similarity`: **0.0** (flat). Latency delta: **-75.6 ms** (candidate faster), but `meaningful_program_rate` stayed at **0.0** on both arms.

## Blocker: local AgentV SDK / NODE_OPTIONS gap (not a repo code bug)

The multi-suite `--ship-gates` eval subprocess (`scripts.evaluate_model ... --ship-gates`) failed for both arms with:

```
RuntimeError: AgentV SDK is unavailable; run npm ci in the checkout or set AGENTV_RUNNER
```

Root cause: this sandbox exports `NODE_OPTIONS="--import tsx" --max-old-space-size=8192` in the ambient shell environment. Node rejects `--import` inside `NODE_OPTIONS` (`node: --import tsx is not allowed in NODE_OPTIONS`, exit 9), so **every** plain `node` invocation fails in this shell, including `npm ci` and the AgentV runner (`scripts/run_agentv_eval.mjs`) that `scripts.evaluate_model` shells out to. The single-suite smoke diagnostic that produced the metrics above ran through a different in-process path and did complete; only the ship-gate AgentV publish step (which writes `scoreboard.json`) was blocked.

Fix applied this cycle: `env -u NODE_OPTIONS npm ci` in the repo root, which installed `node_modules/@agentv/core@4.42.4` and the `agentv` CLI successfully. This is **not** a repository code change (`node_modules/` is untracked) and no `HarnessSignalV1` was raised naming a canonical harness family, so no `repair_harness` handoff action applied — this is local environment setup, not a harness fix.

Residual risk: any future `node` subprocess call from this driver must run with `NODE_OPTIONS` unset (or from a shell without the ambient sandbox override) or the ship-gate AgentV publish step will fail again with the same `missing_scoreboard` symptom.

## SDLC Phase A classification

`positive: false`, `stack_layer: false`, `action: no_stack_layer_non_positive`.

Reasons (from `sdlc_delivery.json`):

1. `measurement_incomplete:c1-control:missing_scoreboard`
2. `measurement_incomplete:c1-bounds:missing_scoreboard`
3. `primary_metric_null_or_worse:smoke.structural_similarity:control=0.0575 candidate=0.0575 improvement=0.0`

The observed latency improvement is a pure latency blip with no accompanying quality win (`meaningful_program_rate=0.0` on both arms, well under the ~1/3 threshold required for a latency-only win) — **not positive** per `sdlc` autotrain-iteration-delivery quality-aware classification, independent of the measurement-completeness gap.

## Next-run priorities

1. **infrastructure:** replay the identical frozen control/candidate arm with the AgentV SDK now installed and `NODE_OPTIONS` unset for the driver subprocess, to complete the missing `scoreboard.json` measurement before testing a new hypothesis.
2. **model:** re-test `grammar_completion_bounds` only once (1) yields a complete scoreboard; the current -75.6 ms delta with `mpr=0.0` on both arms is not model evidence either way.
3. **evaluation:** keep ship gates honest and fail-closed on fixture `n` / `mpr=0.0`; do not weaken for continuous smoke.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-control/`, `.../runs/c20260802-continuous-openui-local-8c0b60dd-c1-bounds/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c1-results.json`
