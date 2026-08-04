# Continuous autotrain resumed-session cycle 3 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

> See [`continuous-openui-20260802-resume-c2-results.md`](continuous-openui-20260802-resume-c2-results.md)
> for why this resumed session's `campaign_id` namespace collides with the earlier same-day
> `continuous-openui-20260802-c3-results.md`. This doc covers a distinct campaign run.

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c3` |
| Source | `26b913a71f2591d2094440aa8a8c62e1100cdd60` |
| Device | CPU |
| Steps | 20 |
| Params (both arms) | 1,755,764 |
| Lever tested | `component-plan` |

## Run matrix

| Arm | smoke n | parse_rate | meaningful_program_rate | structural_similarity | binder_reference_f1 | latency_ms_p50 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| c3-control | 3 | 1.0 | 0.333 | 0.2308 | 0.733 | 3257.35 | ship gates fail (insufficient n) |
| c3-component-plan | 3 | 1.0 | 0.0 | 0.1725 | 0.633 | 3117.64 | ship gates fail (insufficient n + quality floors) |

## What this confirms

The size-matched `component-plan` lever (identical `trainable_params` = 1,755,764 vs control)
**regressed on every quality axis** against the matched control:

- `structural_similarity`: `-0.0583` (0.2308 → 0.1725)
- `meaningful_program_rate`: `0.333 → 0.0`
- `binder_reference_f1`: `0.733 → 0.633` (flagged `non_regression_fail`)

This is a clean **negative screening result** — the driver rejected the `component-plan` lever
for promotion. `fixture_insufficient_n_alone` also applies (smoke `n=3 < 20`), but the primary
metric regression alone is sufficient to reject this lever; it is not just a fixture artifact.
Correctly classified **non-positive**: no stack layer.

## Next-run priorities

1. **model:** the completed non-positive `component-plan` arm is exhausted; test the distinct
   size-matched `component-edge` lever next (`c3-component-edge`).
2. **evaluation:** keep the matched control as the size-matched baseline every cycle.
3. **infrastructure:** soft ship-gate fails on fixture `n` never stop the continuous loop.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c3/`
- JSON twin: `continuous-openui-20260802-resume-c3-results.json`
