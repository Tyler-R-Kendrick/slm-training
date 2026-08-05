# Continuous autotrain cycle 1 results (2026-08-05, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1` |
| Source | `bdf143cd` (current `main`, immediately after #1444) |
| Train | `wf_smoke_v2`, 20 steps / seed 100001 |
| Eval | `e938_role_safe_all_targets_v2` |

## Context: first cycle after the `generate_batch_size` knob unblock

#1444 (`bdf143c`) registered the `generate_batch_size` knob that
`run_autotrain_continuous._matrix()` bakes into every `role="screening"`
hypothesis, which had been failing `HypothesisMatrix` strict validation with
`extra_forbidden` on every prior continuous-openui-local cycle. This is the
first cycle run directly against that fix: the campaign completed a full
measurement with no knob-validation rejection, confirming the unblock holds.

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 4480.77 |
| bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.0575 | 3961.09 |

Primary metric delta (bounds − control) on `smoke.structural_similarity`:
**0.0** (exact tie), matching every prior measurement of this lever.

## Diagnostics

1. `grammar_completion_bounds` again produced an exact `structural_similarity`
   tie against the matched control (0.0575, parse 1.0, meaningful 0.0,
   binder F1 0.6333) — the same null result seen in PR #1322 (c1/c3) and the
   `continuous-openui-local-r2` c1 run (#1431).
2. Latency this run is 11.6% **faster** for `bounds` (3961.09 vs 4480.77 ms
   p50), a second sign reversal at fixture n=3 scale (the r2-c1 run reversed
   the original PR #1322 direction; this run reverses again). Taken with the
   prior two measurements, this confirms retry-to-retry latency deltas on
   this fixture are CPU-sandbox timing noise, not an attributable lever
   effect.
3. Both arms fail honest ship gates on evidence volume alone
   (`smoke:insufficient_n actual=3 need>=20`) — expected at screening scale,
   not a regression.

## Next-run priorities

1. Treat `grammar_completion_bounds` vs matched control as exhausted for
   screening: three independent measurements now show an exact
   `structural_similarity` tie and non-attributable latency noise. Do not
   re-run this pairing without a new preregistered hypothesis.
2. Test the size-matched `component-plan` quality hypothesis next per the
   driver's ranked priorities.
3. Do not promote or ship either checkpoint; both remain screening-scale
   fixture artifacts.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1/`
- JSON twin: `continuous-openui-20260805-8c0b60dd-c1-results.json`
