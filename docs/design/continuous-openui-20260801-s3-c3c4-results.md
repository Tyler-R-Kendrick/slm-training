# Continuous autotrain cycles 3-4, third session — steps-lever disagreement resolved (2026-08-01)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-20260801` (third session; see [c1c2](continuous-openui-20260801-s3-c1c2-results.md)) |
| Campaigns | `continuous-loop-20260801-c3`, `continuous-loop-20260801-c4` |
| Source | `e2df9c52e6e226be9bc5a1e1d0e1fc65d9ab2e05` |
| Device | CPU |

## Headline: the steps-lever disagreement across #1246/#1247/#1248/#1250 is now resolved

Cycle 4 replays the disputed "steps=21 (control) vs steps=42 (candidate)"
`held_out` comparison from
[session 2](continuous-openui-20260801-s2-c4-results.md) and
[the harness-fix writeup](model-build-partial-suite-completion-gate-20260801.md).
This is the **first observed run where both arms fully complete `held_out`**
(5/5 documents each, not 1/5):

| Arm | Levers | `held_out` completed | `structural_similarity` |
| --- | --- | ---: | ---: |
| c4-control | steps=21 | 5/5 | 0.38248 |
| c4-steps | steps=42 | 5/5 | 0.37006 |

Delta: **-0.0124** — a real regression, numerically identical to PR
[#1247](https://github.com/Tyler-R-Kendrick/slm-training/pull/1247)'s measurement
(which also reported `control=0.38248`).

**Conclusion:** #1247's regression finding was correct. #1248's "positive"
claim and session 2's numerically-matching replication were both artifacts of
the control arm completing only 1/5 `held_out` documents in those runs, which
gave an artificially low control value (`0.3417`) that made `steps=42` look
like an improvement. Doubling `steps` does **not** improve
`held_out.structural_similarity` at this recipe — it is a small real
regression once both arms are measured on a comparable, fully-completed
basis.

**Recommendation:** #1246 and #1248 should not be merged as steps-lever
positive evidence. #1247 is the trustworthy result.

## Cycle 3 (screening, non-positive)

| Arm | Levers | completed | structural_similarity | latency_ms_p50 |
| --- | --- | ---: | ---: | ---: |
| c3-control | (none) | 3/3 | 0.1725 | 7102.07 |
| c3-both | grammar_completion_bounds + compact_active_canvas | 3/3 | 0.1725 | 6948.89 |

153ms latency win, zero quality movement — `latency_win_rejected_low_mpr`,
consistent with session 2's c3/c6/c7 findings on the same lever pair.

## SDLC Phase A

Both cycles: `positive=False`, `stack_layer=False`,
`action=no_stack_layer_non_positive`. Docs-only, local commit — no new
stacked layer. Cycle 4 is a genuine, important finding (it resolves an
open cross-PR disagreement) but is not itself a positive lever result, so it
does not earn a stack layer on its own merits — it earns a docs update and,
separately, a recommendation against merging #1246/#1248 as currently framed.

## Next-run priorities

1. Do not merge #1246 or #1248 as steps-lever positive evidence.
2. `grammar_completion_bounds` + `compact_active_canvas` remain latency-only
   at this fixture scale; not worth a dedicated matrix without a
   quality-moving lever.
