# Durable baseline interruption (2026-07-15)

The rerun of the six-step baseline again terminated during E1 decoding. The
new durable artifact recorded `active=E1`, `completed=0/1`, and the checkpoint
was present, while no final quality summary existed. This is explicitly
incomplete evidence; no baseline quality claim is made.

The next experiment will use the faster training/loss telemetry loop and avoid
repeating this resource-heavy decode until the evaluation runtime is isolated.
