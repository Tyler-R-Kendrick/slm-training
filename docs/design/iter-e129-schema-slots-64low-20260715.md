# E129 schema/slot 64-step low-weight control — 2026-07-15

E129 holds E127's lower LTR/fidelity weights fixed while extending training to
64 steps, isolating duration from E128's weight change.

| Suite | n | Parse | Placeholder validity | Normalized fidelity | Structural similarity | Reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 1 | 0.0 | 0.0 | 0.0 | 0.1542 | 0.0 |

Loss finished at **9.89** and telemetry was persisted, but the E127
placeholder signal was not reproduced. This rejects “train longer” as the
current lever and shows E127 is not enough evidence for promotion; the next
iteration must examine data composition and multi-example variance.
