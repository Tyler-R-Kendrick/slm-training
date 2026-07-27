# DSH5-03 exact bulk crossover preflight

- Verdict: **unavailable** — exact compiler fixtures are wiring evidence only; the compatible typed policy is rejected and no matched five-arm serving rows exist
- Recipe: local CPU compiler fixtures at fanout 1/2/4/8; no checkpoint, remote/HF workload, or human-rating gate.
- Primary serving metric remains target-model forward equivalents; it is unmeasured here because every matched policy arm is absent.

| fanout | legal bulk actions | replay exact | primitive equivalent | claim |
| ---: | ---: | --- | --- | --- |
| 1 | 1 | yes | yes | wiring only |
| 2 | 2 | yes | yes | wiring only |
| 4 | 4 | yes | yes | wiring only |
| 8 | 8 | yes | yes | wiring only |

These are real pack/legal-set/executor/replay checks, but they are compiler-fixture wiring evidence only. The DSH3 typed policy is rejected and this run has no primitive-policy, bulk-policy, bulk-disabled, full-generation, or oracle-sequence serving rows. Therefore it does not claim quality, efficiency, a crossover, a model improvement, or ship readiness.
