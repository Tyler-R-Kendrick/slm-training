# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c4` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps | 21 (control) vs 42 (candidate, doubled-steps lever) / batch 2 / seed 7 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Cycle role | promotion |

## Run matrix

| Arm | Levers | held_out n | completed | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c4-control | steps=21 | 3 | 1/3 | 1.0 | 0.0 | 0.3417 | 14313.98 |
| c4-steps | steps=42 | 3 | 3/3 | 1.0 | 0.3333 | 0.37006 | 14783.49 |

Primary delta (steps − control) held-out structural_similarity: **+0.0284**
(0.3417 → 0.37006).

## Diagnostics

1. Doubling train steps (~21 → ~42, still well inside the 3-minute wall cap —
   control finished training in 3.54s, candidate in 5.71s) raised held-out
   `structural_similarity` from 0.3417 to 0.37006.
2. On the smoke suite the same lever also raised `meaningful_program_rate`
   from 0.0 to 0.333 and `structural_similarity` from 0.4167 to 0.51 — the
   **first cycle this session where a lever moved `meaningful_program_rate`
   off the floor**.
3. Phase A classified this **positive** (`primary_metric_win`, quality-aware),
   but `action=positive_no_tracked_delta_skip_stack`: the win is a training
   recipe/knob finding, not a code change, so the driver itself opens no PR.
   This doc write-up is the tracked delta for the cycle.
4. Both arms hit `fixture_insufficient_n` (smoke/held-out n=3); `c4-control`
   also only completed 1/3 held-out documents inside the wall cap (`c4-steps`
   completed 3/3). Ship gates still fail on n/volume — **screening-quality
   signal, not a ship claim**.

## SDLC Phase A

`positive=True`, `stack_layer=False`, `action=positive_no_tracked_delta_skip_stack`.
Because the driver produced no tracked code delta, this markdown + JSON pair
*is* the tracked delta for the positive result; it is bundled with cycles
1–3's docs into a single docs-only commit and pushed as this iteration's
stacked layer (positive result present in the batch).

## Next-run priorities

1. Re-run `steps=42` vs `steps=21` at a larger held-out `n` (above the
   insufficient-n floor) to confirm the `structural_similarity` and
   `meaningful_program_rate` gains are not sampling noise.
2. `c4-control` only completed 1/3 held-out documents in the wall cap while
   `c4-steps` (double the steps) completed 3/3 — re-check whether control-run
   variance (not step count) explains the incompleteness before crediting
   the steps lever with the full delta.
3. If the `meaningful_program_rate` gain replicates, prioritize a dedicated
   steps-budget matrix (`running-experiment-matrices`) over further
   single-cycle continuous screening.

## Artifacts

- Campaign (ephemeral, not committed): `continuous-loop-20260801-c4/`
- JSON twin: `continuous-openui-20260801-c4-results.json`
