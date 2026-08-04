# Continuous autotrain cycle 4 results (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` |
| Campaign | `continuous-loop-20260801-c4` |
| Source | `c1c4eca349b66f05684975575a3640ced50051ea` |
| Device | CPU |
| Steps | 20 (baseline) |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |
| Cycle role | `promotion` (primary metric = `held_out.structural_similarity`) |

## Run matrix

| Arm | smoke latency_ms_p50 | smoke structural_similarity | held_out latency_ms_p50 | held_out structural_similarity | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| c4-control | 19807.48 | 0.4167 | 20777.56 | 0.38248 | eval completed; ship gates fail (insufficient n) |
| c4-steps | 11153.18 | 0.51 | 11075.73 | 0.37006 | eval completed; ship gates fail (same) |

Primary delta (steps − control) `held_out.structural_similarity`: **-0.0124**
(worse; this is the promotion-role primary metric).

## Diagnostics

1. The `steps` candidate roughly **halved latency** on both suites
   (held_out p50 20777.56ms → 11075.73ms) and improved smoke
   `structural_similarity` (0.4167 → 0.51).
2. But this cycle's role is `promotion`, whose primary metric is
   `held_out.structural_similarity`, and that metric moved the wrong
   direction (0.38248 → 0.37006). Per the quality-aware tradeoff policy, a
   latency win never substitutes for a primary-metric regression on a
   promotion-role cycle — classified **non-positive**.
3. Both arms still fail ship gates on fixture `insufficient_n`, expected at
   this scale.

## Next-run priorities

1. **model:** re-screen the `steps` lever under a `screening`-role cycle
   (primary metric = latency, with parse/mpr held) to characterize the
   latency win on its own terms, separate from this promotion-role read.
2. **model:** re-run `steps` at a larger `held_out` suite size before trusting
   the `structural_similarity` delta — n is fixture-scale here.
3. **process:** do not promote on this cycle regardless of the latency
   improvement; primary metric regressed.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-c4/`
- JSON twin: `continuous-openui-20260801-c4-results.json`
