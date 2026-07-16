# E128 schema/slot 64-step iteration — 2026-07-15

E128 extends E127 to 64 CPU steps and doubles both LTR and fidelity loss
weights. It is a negative matched diagnostic.

| Suite | n | Parse | Placeholder validity | Normalized fidelity | Structural similarity | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 1 | 0.0 | 0.0 | 0.0 | 0.1542 | 0.0 |

Loss finished at **15.03**, with 9.89s of persisted training telemetry. The
recipe regressed the placeholder signal seen in E127 and did not improve
syntax. The higher loss weights are rejected; E127's lower-weight conditioning
recipe remains the better hypothesis for the next data-composition test.
