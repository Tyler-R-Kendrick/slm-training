# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c4` |
| Source | `1bdfb14ebcf2393976a7c969e7bdd449fc5ada39` |
| Device | CPU |
| Steps (configured) | 20 (control) vs 40 (candidate, doubled-steps lever) |
| Steps (observed) | 21 (control) vs 42 (candidate) — harness runs one extra step past the configured target |
| Batch / seed | 2 / 7 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Cycle role | promotion |

## Run matrix

Two separate suites ran this cycle: `smoke` (n=3) and `held_out` (n=5, the
`primary_metric` suite). Do not read the two rows for one arm as the same
population.

| Arm | Levers | Suite | n | completed | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| c4-control | steps=21 | smoke | 3 | 1/3 | 1.0 | 0.0 | 0.4167 | 14313.98 |
| c4-control | steps=21 | held_out | 5 | 1/5 | 1.0 | 0.0 | 0.3417 | 19067.94 |
| c4-steps | steps=42 | smoke | 3 | 3/3 | 1.0 | 0.3333 | 0.51 | 14783.49 |
| c4-steps | steps=42 | held_out | 5 | 5/5 | 1.0 | 0.0 | 0.37006 | 14541.95 |

Primary delta (steps − control) **held-out** structural_similarity: **+0.02836**
(0.3417 → 0.37006). Held-out `meaningful_program_rate` stayed 0.0 on both arms.

## Diagnostics

1. Doubling train steps (~21 → ~42, still well inside the 3-minute wall cap —
   control finished training in 3.54s, candidate in 5.71s) raised held-out
   `structural_similarity` from 0.3417 to 0.37006 (+0.02836). Held-out
   `meaningful_program_rate` did **not** move (0.0 on both arms).
2. On the separate smoke suite the same lever raised `meaningful_program_rate`
   from 0.0 to 0.333 and `structural_similarity` from 0.4167 to 0.51 — the
   **first cycle this session where a lever moved `meaningful_program_rate`
   off the floor** — but that gain is scoped to smoke only, it did not
   replicate on held-out.
3. Phase A classified this **positive** on `held_out.structural_similarity`
   (`primary_metric_win`, quality-aware), but
   `action=positive_no_tracked_delta_skip_stack`: the win is a training
   recipe/knob finding, not a code change, so the driver itself opens no PR.
   This doc write-up is the tracked delta for the cycle.
4. Both suites hit `fixture_insufficient_n` (smoke n=3, held-out n=5);
   `c4-control` also only completed 1/5 held-out documents inside the wall
   cap (`c4-steps` completed 5/5). Ship gates still fail on n/volume —
   **screening-quality signal, not a ship claim**.

## SDLC Phase A

`positive=True`, `stack_layer=False`, `action=positive_no_tracked_delta_skip_stack`.
Because the driver produced no tracked code delta, this markdown + JSON pair
*is* the tracked delta for the positive result; it is bundled with cycles
1–3's docs into a single docs-only commit and pushed as this iteration's
stacked layer (positive result present in the batch).

## Next-run priorities

1. Re-run `steps=42` vs `steps=21` at a larger held-out `n` (above the
   insufficient-n floor) to confirm the held-out `structural_similarity` gain
   is not sampling noise.
2. `c4-control` only completed 1/5 held-out documents in the wall cap while
   `c4-steps` (double the steps) completed 5/5 — re-check whether control-run
   variance (not step count) explains the incompleteness before crediting
   the steps lever with the full held-out delta.
3. The smoke-suite `meaningful_program_rate` gain (0.0→0.333) did not carry
   over to held-out (mpr stayed 0.0 on both arms); if it does not replicate
   on held-out at larger n, treat it as a smoke-fixture artifact rather than
   a general lever before prioritizing a dedicated steps-budget matrix
   (`running-experiment-matrices`).

## Artifacts

- Campaign (ephemeral, not committed): `continuous-loop-20260801-c4/`
- JSON twin: `continuous-openui-20260801-c4-results.json`
