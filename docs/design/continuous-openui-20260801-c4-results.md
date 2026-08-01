# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260801-c4` |
| Source | `24c20769c366aeb9e9f7a98eb72089b3a97859c7` |
| Device | CPU |
| Steps | control 20 / candidate 60 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Primary metric | `held_out.structural_similarity` (**unavailable** — suite missing) |

## Run matrix (smoke, informational only — primary metric was unavailable)

| Arm | Steps | smoke n | parse_rate | meaningful_program_rate | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| c20260801-c4-control | 20 | 3 | 1.0 | **0.667** | 11879.46 | eval completed; ship gates fail (insufficient n) |
| c20260801-c4-steps | 60 | 3 | 1.0 | 0.333 | 20430.57 | eval completed; ship gates fail (insufficient n) |

Classified **non-positive**: `primary_metric_unavailable` (held_out suite missing)
takes precedence over any smoke-only delta.

## Diagnostics

1. The declared primary metric for this cycle, `held_out.structural_similarity`,
   has no data — `e938_role_safe_all_targets_v2` does not publish a `held_out`
   suite, matching the `missing_suite` pattern already seen for
   `held_out`/`adversarial`/`ood`/`rico_held` in earlier cycles this loop.
   This is an eval-snapshot binding gap, not a model result.
2. On smoke (informational): control's `meaningful_program_rate` reached
   0.667 (2/3), the best of the four cycles run today (c1-c3 were all 0.0).
   Not yet attributable to a lever — no lever changed between control here and
   the 0.0 controls in c1-c3; likely a seed/sample-order effect at n=3.
3. Raising steps from 20 to 60 under the same 3-minute wall cap regressed
   both quality (0.667 → 0.333) and latency (11879 → 20431 ms) — evidence
   the extra steps are pushing runs toward wall-cap truncation rather than
   buying more signal at this scale.

## Next-run priorities

1. **infrastructure/eval:** bind an `eval_version`/`eval_suites` combination
   that actually publishes a `held_out` suite before using
   `held_out.structural_similarity` as a continuous primary metric, or switch
   the continuous primary metric to a suite that is present (`smoke`).
2. **model:** investigate whether control's 0.667 `meaningful_program_rate`
   reproduces on a repeat run with the same seed before treating it as signal.
3. **model:** do not raise `steps` past the point where the wall cap starts
   truncating decode — 60 steps degraded both quality and latency vs 20 steps
   under the same 3-minute cap.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c4/`
- JSON twin: `continuous-openui-20260801-c4-results.json`
