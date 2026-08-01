# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim, not promoted.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c4` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | 20 (control) vs. doubled (candidate) |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 12 minutes (promotion-role campaign) |
| Hypothesis | Doubling steps without levers only raises cost and does not improve unit decode latency |

## Run matrix

| Arm | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | --- |
| c4-control | 1.0 | 0.333 | 20780.20 | eval completed; ship gates fail (insufficient n) |
| c4-steps (doubled steps) | 1.0 | 0.333 | 11447.07 | eval completed; ship gates fail (insufficient n) |

Latency change (steps − control): **−9333.13 ms** (candidate ~45% faster),
`meaningful_program_rate` and `parse_rate` **held identical** — a
quality-held latency win.

## Diagnostics

1. **The stated hypothesis is refuted.** It predicted doubling steps would
   only raise cost with no latency benefit; the observed candidate was
   substantially *faster* at held quality, not slower or unchanged.
2. The campaign's preregistered primary metric
   (`held_out.structural_similarity`) was not measured this cycle
   (`primary_metric_unavailable`), so positivity here rests on the
   secondary `mpr_per_ms` (meaningful-program-rate-per-millisecond)
   efficiency heuristic, which improved ~81.5% (1.604e-05 → 2.912e-05).
   Treat this as a provisional signal, not a confirmed win, until the
   primary metric is actually measured on a replay.
3. Fixture n is still insufficient on both arms (expected at smoke scale) —
   the ship gates correctly fail regardless of the latency finding.
4. The driver's own SDLC Phase A classifier marked this cycle
   `positive=true` but `stack_layer=false`
   (`positive_no_tracked_delta_skip_stack`): there is no code/harness delta
   behind this result (it is a steps-count knob explored within the
   campaign, not a tracked change), so per `autotrain-iteration-delivery`
   no stacked PR is opened — the finding lands as a docs-only local commit.

## Next-run priorities

1. **evaluation:** re-run the steps-doubling arm with `held_out` bound so
   the preregistered primary metric is actually measured.
2. **model:** replicate at a larger n before treating this as more than a
   provisional efficiency signal; do not promote from fixture n.
3. **delivery:** no stack layer for this cycle (no tracked code/harness
   delta) — matches the driver's own classification.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c4/` (local only, gitignored)
- Runs: `.../runs/c20260801-c4-control/`, `.../runs/c20260801-c4-steps/`
- SDLC delivery record: `outputs/autoresearch/continuous-loop-20260801-c4/sdlc_delivery.json` (`positive=true`, `stack_layer=false`, `action=positive_no_tracked_delta_skip_stack`)
- JSON twin: `continuous-openui-20260801-c4-results.json`
