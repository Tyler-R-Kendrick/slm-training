# E285 — exact-signature profile aborted

Date: 2026-07-17
Status: **aborted; invalid evidence**

E285 attempted a read-only exact-decision-signature gradient profile using the
frozen E228 checkpoint and committed E283 preference corpus. The intended recipe
matched E284 except for `train_strata=decision_signature`.

The direct diagnostic invocation had no experiment-level wall-clock deadline and
remained incomplete beyond 25 minutes. It was operator-stopped, produced no report,
and changed no optimizer state or checkpoint. No metric, comparison, or training
decision may be inferred from this attempt.

## Remediation

The shared autoresearch harness now exposes `CampaignBudget.max_wall_minutes` as a
configurable positive value with a five-minute default and hard maximum. The
deadline is cumulative: data build, training, and evaluation consume the same
budget rather than each receiving a fresh timeout. Expiry records a stopped
`ExperimentOutcome` with `wall_time_budget_seconds`.

Future experiment comparisons must use the same declared training budget. A larger
step, token, sample, or wall budget is a separate scaling experiment, not evidence
that one lever beats another.

Machine-readable record:
[quality-matrix-v10-e285-exact-signature-aborted-results.json](quality-matrix-v10-e285-exact-signature-aborted-results.json).
