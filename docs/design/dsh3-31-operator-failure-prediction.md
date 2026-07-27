# DSH3-31: early operator-policy failure prediction (SLM-406)

The shadow-only predictor consumes only the evaluator's bounded per-record
prefix-time DecodeStats projection: normalized position, legal candidate count,
and forced-choice state.  Final parse/meaningful-program verdicts, errors,
decode outcome, stop reason, timeout, fallback, and compiler termination fields
are labels or diagnostics and are rejected as feature leakage.

Rows are grouped by request/target cluster and checkpoint.  Any future
train/validation/test split must keep each group intact.  The preflight refuses
to run a predictor until the canonical eval evidence has per-record temporal
rows; aggregate DecodeStats are not trajectory-safe evidence.

This issue does not change production decode routing.  Any threshold remains a
shadow abort/defer simulation until it clears a group-safe held-out comparison
against the entropy/margin baseline with false-abort cost reported.

## Local evidence (2026-07-26)

`dsh3-31-operator-failure-prediction-20260726-local/report.json` records an
unavailable preflight, not a model result: the only locally discovered
playground checkpoint failed the required `symbol_only/v2` contract before
generation, and a transient test checkpoint was no longer present.  Therefore
there are zero temporal rows, no classifier fit, no AUROC/AUPRC/calibration or
false-abort measurement, and no ship or fixture-performance claim.  The
canonical evaluator now preserves leak-sealed per-record prefix evidence so a
compatible local evaluation can make the planned comparison without rerouting
decode.
