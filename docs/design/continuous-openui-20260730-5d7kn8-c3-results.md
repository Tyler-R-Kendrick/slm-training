# Continuous autotrain cycle 3 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260730` |
| Campaign | `continuous-loop-20260801-c3` |
| Predecessor | `continuous-loop-20260730-c2` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 40 / batch 2 / seed 7 (2x cycle 2's step budget) |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` (bound explicitly per cycle 2 priority 1) |
| Wall cap | 3 minutes |
| Version stamp | `code_commit=311db12a`, `harness.autoresearch.experiment_campaign=v26`, `harness.model_build.eval=v71` |

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| c20260801-c3-control | bounds off, canvas off | 3 | 1.0 | 0.333 | 3686.62 | eval completed; ship gates fail (insufficient n) |
| c20260801-c3-both | bounds **on**, canvas **on** | 3 | 1.0 | 0.333 | 3710.51 | eval completed; ship gates fail (same) |

Primary delta (both − control) p50 latency: **23.89 ms** (positive = levers slower).

## Diagnostics

1. Binding `eval_version=e938_role_safe_all_targets_v2` explicitly (per cycle 2
   next-priority 1) kept both arms on the default-safe path — no repeat of the
   `v1` missing-suite failure.
2. At 40 steps, `meaningful_program_rate` rose to 0.333 for both arms (from
   0.0 at 20 steps in cycle 2), and the control/candidate delta was
   **unchanged** (0.333 → 0.333, Δ=0.0, not a missing/unavailable value) —
   `grammar_completion_bounds` + `compact_active_canvas` together still did
   not improve smoke p50 latency or mpr at the larger step budget. Only the
   combined arm was tested this cycle; `compact_active_canvas` has not been
   isolated from `grammar_completion_bounds` in any cycle so far (cycle 2
   tested `grammar_completion_bounds` alone against control with the same
   null mpr outcome).
3. First attempts this cycle failed closed with `RuntimeError: AgentV SDK is
   unavailable; run npm ci in the checkout or set AGENTV_RUNNER` — the
   sandbox checkout had `npm ci` run under `src/apps/openui_bridge` and
   `src/apps/design_md_bridge` but not at the **repo root**, where
   `@agentv/core` is pinned (`package.json`). README already documents `npm
   ci` before eval commands (line 311); this was a one-time environment setup
   gap, not a code defect, so no stack layer was opened for the unblock.

## Next-run priorities

1. **model:** do not promote `grammar_completion_bounds` + `compact_active_canvas`
   together on latency alone; mpr stayed unchanged between control and the
   combined candidate at both 20 (cycle 2) and 40 (cycle 3) steps. Do not
   attribute this to `compact_active_canvas` independently — it has never
   been tested in isolation.
2. **data:** re-test bounds/canvas on a `train_version` with more synthesis
   volume so smoke `n` clears the fixture-insufficient-n gate and deltas
   become a real comparative signal (current n=3 is wiring-only).
3. **evaluation:** keep ship gates honest; do not weaken for continuous
   smoke; fixture `insufficient_n` fails are expected, not terminal.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c3/`
- Runs: `.../runs/c20260801-c3-control/`, `.../runs/c20260801-c3-both/`
- SDLC delivery: `.../sdlc_delivery.json` (`positive=false`,
  `stack_layer=false`, `action=no_stack_layer_non_positive`)
- JSON twin: `continuous-openui-20260730-c3-results.json`
