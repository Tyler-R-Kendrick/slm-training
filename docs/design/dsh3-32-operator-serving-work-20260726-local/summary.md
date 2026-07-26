# DSH3-32 operator-serving work preflight

- Verdict: **unavailable** — required arms are not all measured under one serving identity
- Recipe: local CPU evidence inspection; no remote workload, checkpoint, or human-rating gate.
- Primary metric: target-model forward equivalents, reconciled from raw target and calibrated non-target forwards.
- Typed-policy quality preflight: `reject` — no typed head caused a beneficial enabled-versus-zero held-out choice change
- Measured serving rows: 0

This is not a ship or efficiency claim. Every serving arm must share the recorded identity and matched meaningful-parse quality; failures, timeouts, legal-set construction, dry runs, executor, validation, materialization, caching, batching, CPU/device, and wall work stay in the denominator.
