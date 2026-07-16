# E115 small model with E91 settings (2026-07-15)

E115 tested whether E91's stronger settings could recover quality on the small
128-dimensional model: frozen context, structural bias `2.5`, mixed masking,
schema context, LTR weight `2.0`, fidelity weight `1.5`, and 16 generation
steps.

The process terminated at step `7` without a checkpoint or summary. Partial
telemetry is preserved in
`outputs/runs/iter-e115-small-e91-settings-20260715/e115_small_e91_settings/metrics.jsonl`.
No evaluation or quality claim is made.

Decision: incomplete/resource-terminated. The higher-quality settings trigger
the same environment failure even at small model size; use the known-completing
small recipe while instrumenting resource/exit behavior.
