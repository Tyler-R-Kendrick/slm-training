# Continuous autotrain loop `gdkj7n31`, cycles c5-c6 (2026-08-01, round 2)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Second round of the same session's continuous loop (env already warm: venv +
`npm ci` done, `NODE_OPTIONS` cleared — see
[cycles c1-c4](continuous-openui-gdkj7n31-c1-c4-results.md) for the env-gap
diagnosis and self-heal).

| Field | Value |
| --- | --- |
| Loop | `gdkj7n31` |
| Upstream commit | `41d874c76b9ed68f4c6d375366ea4398b95a0429` |
| Device | CPU |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes / arm |

## Run matrix

| Cycle | Arm | smoke n | parse_rate | mpr | latency_ms_p50 | structural_similarity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| c5 | control | 3 | 1.0 | 0.0 | 3285.26 | 0.0563 | eval completed; ship gates fail (insufficient n) |
| c5 | batch1 | 3 | 1.0 | 0.0 | 18549.83 | 0.0184 | eval completed; ship gates fail — **much worse latency** |
| c6 | control | 3 | 1.0 | 0.0 | 2515.23 | 0.0964 | eval completed; ship gates fail (insufficient n) |
| c6 | bounds | 3 | 1.0 | 0.0 | 2379.55 | 0.0964 | eval completed; ship gates fail (small latency win, mpr-gated) |

Both cycles classified **non-positive** (SDLC Phase A) — no stack layer opened.

## Diagnostics

1. **c5 — `batch_size=1` finding:** at this fixed 20-step budget, dropping to
   `batch_size=1` made smoke p50 latency ~5.6x worse (3285ms → 18550ms) and
   `structural_similarity` worse (0.056 → 0.018). Reads as more, smaller
   optimizer steps costing more wall time for the same fixed step count on
   CPU at fixture scale — not a harness bug, just not worth pursuing at this
   budget.
2. **c6 — repeats the c1 `grammar_completion_bounds` pattern:** a small smoke
   p50 latency win (135.7ms) again shows up, and is again rejected because
   `meaningful_program_rate=0.0` sits below the 1/3 floor required for a
   latency-only win to count (quality-aware tradeoff gate in
   `run_autotrain_continuous._classify_metric_tradeoff`). This lever needs a
   non-fixture run (real `mpr` signal) before its quality effect is
   measurable — not evidence the lever itself is broken.

## Next-run priorities

1. Do not pursue `batch_size=1` for latency at fixture scale (c5 finding).
2. Re-test `grammar_completion_bounds` once a non-fixture run can produce
   `meaningful_program_rate > 0` signal, to see if the small latency win holds
   under a real quality-held comparison.
3. Merge [#1254](https://github.com/Tyler-R-Kendrick/slm-training/pull/1254)
   (AgentV `NODE_OPTIONS` fix, still open).

## Artifacts

- Campaigns: `outputs/autoresearch/continuous-loop-20260801-c{5,6}/`
  (gitignored; not part of this PR)
- JSON twin: `continuous-openui-gdkj7n31-c5-c6-results.json`
