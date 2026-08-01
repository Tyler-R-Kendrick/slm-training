# Continuous autotrain cycles 1-3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260801-c3` (cycles 1-2 were environment-screening failures under the same loop) |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 20 / batch 2 / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Environment repairs (cycles 1-2, no code changes)

The scheduled-run execution container had neither a Python 3.12 venv with
`torch`, nor the AgentV Node SDK installed, so the first two cycles failed
closed before any model evidence existed:

| Cycle | Failure | Repair |
| --- | --- | --- |
| c1 (`c20260801-c1-control` / `-bounds`) | `ModuleNotFoundError: No module named 'torch'` in `detect_device` | Created `.venv` (python3.12), installed the CI-pinned `torch==2.5.1+cpu` wheel |
| c2 (`c20260801-c2-control` / `-canvas`) | `RuntimeError: AgentV SDK is unavailable; run npm ci` from `src/slm_training/evals/agentv.py` | Ran `npm ci` (with the session's malformed default `NODE_OPTIONS` unset for the invocation) |

Neither repair touched tracked code; both are recorded here so the next
scheduled cycle does not re-diagnose the same infrastructure gap.

## Run matrix (cycle 3)

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c3-control | bounds off, canvas off | 3 | 1.0 | 0.0 | 6768.01 | eval completed; ship gates fail (insufficient n + quality) |
| c3-both | bounds **on**, canvas **on** | 3 | 1.0 | 0.0 | 6656.24 | eval completed; ship gates fail (same) |

Primary delta (both − control) p50 latency: **-111.77 ms** (bounds+canvas
faster). Rejected as non-positive: `meaningful_program_rate` is `0.0` in both
arms, so the latency-only win has `mpr=0.0 < 1/3` — a speedup with no
meaningful programs is not a quality-aware win (SDLC Phase A
`latency_win_rejected_low_mpr`).

## Diagnostics

1. Cycles 1-2 were pure execution-environment gaps (missing `torch`, missing
   the AgentV Node SDK), not harness or model bugs — repaired without any
   tracked-file change.
2. Once repaired, cycle 3 ran the full `train_model` → `evaluate_model
   --ship-gates` pipeline to completion on both arms.
3. Ship gates correctly fail on quality/volume thresholds
   (`meaningful_program_rate`, `structural_similarity`,
   `component_type_recall`, `ast_beq_rate`, `canonical_beq_rate`, and
   `insufficient_n` at `n=3` vs the required `>=20`) — expected for a
   3-example, 20-step smoke fixture, not evidence of a regression.
4. `grammar_completion_bounds=True` + `compact_active_canvas=True` together
   improved raw p50 latency by 111.77ms, but SDLC Phase A correctly
   classifies this **non-positive** because `mpr=0.0` in both arms.

## SDLC Phase A

`classification=NON_POSITIVE stack_layer=False` — no stacked PR opened for
this cycle set. Docs-only, local-commit delivery per
`autotrain-iteration-delivery`.

## Next-run priorities

1. **infrastructure:** pin `torch==2.5.1+cpu` + `npm ci` into the continuous
   execution environment ahead of cycle 1 so this class of screening failure
   does not repeat.
2. **model:** re-test `grammar_completion_bounds` / `compact_active_canvas`
   once `meaningful_program_rate` clears the mpr floor on a non-3-example
   suite, so a latency delta is attributable to a real quality-preserving
   speedup rather than noise on an already-failing arm.
3. **evaluation:** keep ship gates honest; do not weaken for continuous
   smoke; keep RL locked.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c3/`
- Runs: `.../runs/c20260801-c3-control/`, `.../runs/c20260801-c3-both/`
- JSON twin: `continuous-openui-20260801-c3-results.json`
