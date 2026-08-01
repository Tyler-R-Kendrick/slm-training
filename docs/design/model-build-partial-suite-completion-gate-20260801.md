# Phase A: gate primary-metric comparisons on matched suite completion (2026-08-01)

## Problem

Three parallel `continuous-openui-20260801` autotrain sessions today all ran the
same nominal "steps=21 (control) vs steps=42 (candidate)" cycle against the
`e938_role_safe_all_targets_v2` `held_out` suite under the 3-minute wall cap,
and disagreed on the verdict:

| PR | control `held_out.structural_similarity` | candidate | delta | verdict |
| --- | ---: | ---: | ---: | --- |
| [#1248](https://github.com/Tyler-R-Kendrick/slm-training/pull/1248) | 0.3417 | 0.37006 | +0.0284 | positive |
| [#1247](https://github.com/Tyler-R-Kendrick/slm-training/pull/1247) | 0.38248 | 0.37006 | -0.0124 | regression |
| [#1250](https://github.com/Tyler-R-Kendrick/slm-training/pull/1250) (session 2) | 0.3417 | 0.37006 | +0.0284 | positive (matches #1248) but flagged unreliable |

Session 2's write-up (`continuous-openui-20260801-s2-c4-results.md`, PR #1250)
diagnosed the root cause: the **candidate** arm (`steps=42`) reliably completes
all 5/5 `held_out` documents inside the wall cap in every session
(`held_out.structural_similarity` pinned at `0.37006` in all three). The
**control** arm only completes **1 of 5** documents in two of the three
sessions (`completed_document_n=1`), so its `structural_similarity` average
reflects whichever single document happened to finish — not a 5-document
average — and swings with which document that was. `held_out` p50 latency for
control clusters near 19.1-19.5s in all three sessions, consistent with one
slow document dominating a near-empty completed sample.

`_classify_positive_metrics` (`src/slm_training/autoresearch/climb_policy.py`)
had no gate on this: it compared the raw `held_out.structural_similarity`
averages directly, so a wall-clock completion race — not a real effect of the
`steps` lever — could mint (or reject) a `primary_metric_win`.

## Fix (code)

- `scripts/run_autotrain_continuous.py::_run_metrics` now also captures
  `<suite>.n` and `<suite>.completed_document_n` (namespaced only; never
  merged into the bare quality/latency leaves used by the tradeoff paths).
- `src/slm_training/autoresearch/climb_policy.py::classify_positive_metrics`
  now rejects a primary-metric comparison as
  `primary_metric_incomparable_partial_suite:<metric>:...` — neither a win nor
  a regression — whenever either arm's `completed_document_n < n` for the
  suite named in the primary metric (e.g. `held_out.*` checks
  `held_out.completed_document_n`/`held_out.n`). Fully-completed comparisons
  are unaffected.
- `scripts/run_autotrain_continuous.py::_classify_positive` no longer lets the
  quality/efficiency tradeoff path (`_classify_metric_tradeoff`) override a
  partial-suite rejection back to positive.

Replaying the exact #1247/#1248/session-2 "cycle 4" numbers
(`control: completed_document_n=1/5, structural_similarity=0.3417` vs
`candidate: completed_document_n=5/5, structural_similarity=0.37006`) through
`_classify_positive` now returns `positive=False` with reason
`primary_metric_incomparable_partial_suite:held_out.structural_similarity:...`
instead of `primary_metric_win`. Fully-completed comparisons (both arms
`completed_document_n == n`) are unaffected — regression test asserts a
0.30 → 0.40 fully-completed comparison still classifies positive.

## Tests

- `tests/test_autoresearch/test_climb_policy.py::test_classify_positive_rejects_partial_suite_completion`
- `tests/test_scripts/test_run_autotrain_continuous.py::test_classify_positive_promotion_rejects_partial_held_out_completion`
- `tests/test_scripts/test_run_autotrain_continuous.py::test_run_metrics_loads_held_out_when_preferred` (extended to assert `n`/`completed_document_n` capture)

## Disposition for #1246 / #1247 / #1248 / #1250's steps-lever finding

The post-fix replay is recorded in
[`continuous-openui-20260801-s3-c3c4-results.md`](continuous-openui-20260801-s3-c3c4-results.md):
both arms completed `held_out` 5/5 for the first time, and
`held_out.structural_similarity` moved `0.38248 → 0.37006` (`-0.01242`) — a
real regression, matching #1247's measurement exactly. #1246 and #1248's
"positive" claims were both artifacts of the control arm completing only 1/5
`held_out` documents in those runs; they should not be merged as steps-lever
positive evidence.

## Honesty

This is a harness/measurement-integrity fix (family: `model_build`), not a
training lever result — `fixture_or_scratch` scope, not a ship claim. No gate
was weakened: the change only prevents comparisons that were already
statistically meaningless (averaging one document against five) from being
scored as if they were comparable.
