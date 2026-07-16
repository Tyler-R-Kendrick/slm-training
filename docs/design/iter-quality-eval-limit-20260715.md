# Iteration: bounded quality diagnostics (2026-07-15)

The quality evaluator now supports an explicit `--eval-limit` diagnostic cap
for every selected suite. The default remains unlimited; ship and full-matrix
runs are unchanged. Metrics record `eval_limit` and `diagnostic_subset` so a
bounded result cannot be mistaken for a ship result.

The cap was needed because smoke-only matrix retries were spending the full
decode cost before producing feedback. A one-record E1 probe must be run with
an E0 checkpoint; a direct E1-only invocation correctly fails closed when that
parent checkpoint is absent. The subsequent E0→E1 probe was interrupted by the
environment during E0 decode, but `quality_matrix_progress.json` retained the
active experiment, proving the interruption is observable and resumable.

Validation: `12 passed` in `test_eval_gates.py` and `test_agentv.py`; Python
compileall and `git diff --check` passed.

Follow-up: the matrix runner now handles SIGINT/SIGTERM and writes
`status: interrupted` with the active experiment before exiting. A five-second
forced-termination check produced that state for E0, confirming the fix.
