# Continuous autotrain cycle 1 results (2026-08-05, `continuous-openui-local`, session `dsz5wf`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1` |
| Source | `bdf143cd` (current `main`) |
| Train | `wf_smoke_v2`, 20 steps / seed 100001 |
| Eval | `e938_role_safe_all_targets_v2` |

## Context: independent same-day reproduction

This is a separate scheduled session running the same `continuous-openui-local`
bounds screen already documented in PR #1450's c1 (same recipe, same seed
100001, same source commit `bdf143cd`). It is included here rather than
discarded because it is a genuine independent replicate at fixture scale, and
because it motivated a real harness fix (see below).

## Run matrix

| Arm | Levers | smoke n | parse_rate | meaningful_program_rate | structural_similarity | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| control | bounds off | 3 | 1.0 | 0.0 | 0.0575 | 4306.77 |
| bounds | bounds **on** | 3 | 1.0 | 0.0 | 0.0575 | 5746.75 |

Primary metric delta (bounds − control) on `smoke.structural_similarity`:
**0.0** (exact tie), the 4th independent measurement of this null.

## Diagnostics

1. `grammar_completion_bounds` again produced an exact `structural_similarity`
   tie against the matched control (0.0575, parse 1.0, meaningful 0.0,
   binder F1 0.6333) — consistent with PR #1322 (c1/c3), the
   `continuous-openui-local-r2` c1 run (#1431), and PR #1450's same-day c1.
2. Latency reverses direction again: `bounds` is 33.5% **slower** this run
   (5746.75 vs 4306.77 ms p50), while PR #1450's c1 measured `bounds` 11.6%
   **faster** on the identical recipe/seed. Taken together with the prior
   reversals, this confirms fixture-scale p50 latency deltas on this
   CPU sandbox are timing noise, not an attributable lever effect.
3. Both arms fail honest ship gates on evidence volume alone
   (`smoke:insufficient_n actual=3 need>=20`) — expected at screening scale.

## Harness fix landed this session

Diagnosed why a fresh scheduled session always re-selects the arm bank's
first slug (`bounds`) even when it is already conclusively closed in prior
same-day sessions' open PRs: `_recent_completed_nonpositive_slugs` walks
`predecessor_campaign_id` chains under `outputs/autoresearch/`, which is
gitignored and empty on every fresh checkout. Added an optional
`--skip-slugs` CLI flag to `scripts/run_autotrain_continuous.py`
(`run_cycle(extra_skip_slugs=...)`), unioned into the existing skip-set
computation, so an agent can manually carry forward known-exhausted arm
slugs across sessions. Default (omitted) is behavior-neutral. A durable,
automatic cross-session closure ledger remains open follow-up work.

## Next-run priorities

1. Treat `grammar_completion_bounds` vs matched control as exhausted for
   screening — 4 independent same-day-or-earlier measurements now show an
   exact `structural_similarity` tie with non-attributable, sign-reversing
   latency noise.
2. Use `--skip-slugs bounds,component-plan,component-edge` (or the current
   day's known-exhausted set) on future fresh scheduled sessions until a
   durable cross-session ledger exists.
3. Do not promote or ship either checkpoint; both remain screening-scale
   fixture artifacts.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260805-continuous-openui-local-8c0b60dd-c1/`
- JSON twin: `continuous-openui-20260805-dsz5wf-c1-results.json`
