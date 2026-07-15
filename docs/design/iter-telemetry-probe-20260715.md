# Telemetry probe (2026-07-15)

One CPU scratch training step on the remediated corpus persisted a checkpoint,
`train_telemetry.json`, and deterministic `run_insights.json`. Backward was
69.45% of wall time, followed by forward at 11.11%. The probe is wiring
evidence, not a quality or ship result. The next controlled performance test
should evaluate gradient accumulation or supported mixed precision.
