# E519 — honest slot-contract training context

E519 closes a train/eval authority mismatch. Before this change,
`train_model --slot-contract-in-context` could condition on gold record
placeholders, while honest evaluation derives inventory only from visible
prompt/DESIGN input. The canonical train CLI now exposes
`--honest-slot-contract`, passes the existing fail-closed model setting, and
records both context and honesty flags in the train summary. Train harness
version advances to `v7`.

The matched E519 run differs from E517 only in contract authority. It starts
from committed clean source `950007fc150c73391bd1023928ae0e9e8e1cf065`,
completes 101 CPU HF-context steps / 5,000 target tokens in 103.2 seconds under
`max_wall_minutes=3`, and verifies serving checkpoint SHA
`d82155b03531c2d852ec8d497d3fdb0878ac1f678c0c5d247e272bc36c91805f`
in the OpenUI bucket.

| Arm | Contract authority | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E517 | Gold-derived during training | 0.00 | 0.4083 | 0.2250 | 0.2083 | 0.7445 | 0.2833 | 0.0625 | 0/1 |
| E519 | Visible prompt/DESIGN only | 0.00 | 0.4083 | 0.2250 | 0.2083 | 0.7445 | 0.2833 | 0.0625 | 0/1 |

The checkpoints are not identical: 102/106 tensor entries change, with maximum
absolute delta `9.67e-05`. Nevertheless, E520 exactly matches E518 on every
quality metric and decoder counter. The lower observed latency is unreplicated
runtime variance and is not a performance claim.

## Decision

Retain the honest training path because it removes a privileged-data channel.
Reject E519 for promotion: strict binding-aware meaning and AgentV remain zero.
The next data intervention must make role labels visible in ordinary prompts,
not restore gold placeholder inventory.

Exact recipe, provenance, metrics, and gate outcome:
[machine-readable record](iter-e519-honest-slot-context-20260719.json).
