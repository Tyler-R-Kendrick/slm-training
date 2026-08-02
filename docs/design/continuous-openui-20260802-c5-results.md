# Continuous autotrain cycle 5 results (2026-08-02)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260802` |
| Campaign | `continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c5` |
| Source | `3cb06385d95769bbcde2e5b0b0e0f16138b1d3c8` |
| Device | CPU |
| Steps | 20 |
| Hypothesis | `component-plan` lever (`compiler_decode_mode=tree`, `component_plan_decode_weight=1.0`) |

## Run matrix

| Arm | smoke n | completed docs | decode_timeout_count | Status |
| --- | ---: | ---: | ---: | --- |
| c5-control | 3 | 2 | 1 | measurement incomplete (1 doc timed out) |
| c5-component-plan | 3 | 0 | 3 | measurement incomplete (all 3 docs timed out) |

## Diagnosis

A third, distinct infrastructure signal this loop (after the c2 AgentV SDK gap and the c3
one-run control timeout): every `component-plan` smoke document hit `decode_timeout` within its
~6.93s per-document budget (`effective_decode_timeout_seconds_min≈6.93`, wall cap 3 minutes total
including training). The control arm also lost 1/3 documents to a timeout, suggesting the
constrained-decode path is already close to the wall-budget edge on this container's CPU in
general, and the `component_plan_decode_weight=1.0` profile (all other decode weights zeroed)
pushes it over consistently.

This has **not** been isolated as a pathological bug (e.g. an infinite loop or runaway branching
specific to `compiler_decode_mode=tree` with that weight profile) versus expected CPU-only
slowness for this lever. Blindly retrying the identical arm would just re-timeout and burn wall
budget without new information — a dedicated `repair_harness` (family `model_build`) profiling
pass is needed first.

## Next-run priorities

1. **repair_harness (model_build):** profile `compiler_decode_mode=tree` +
   `component_plan_decode_weight=1.0` to determine bug vs. budget.
2. Do not retry `c5-component-plan` until profiled.
3. **harness (separate, still open):** dedicated repair pass for the 64 pre-existing
   `tests/test_evals` failures on `main` flagged in cycle 2's doc.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-202608-39ee9cf7-c5/`
- JSON twin: `continuous-openui-20260802-c5-results.json`
