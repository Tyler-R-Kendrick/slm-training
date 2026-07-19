# E513 — durable slot-role supervision continuation

E513 warm-starts the durable E396 checkpoint on the 260-row E500 projected
corpus with 50% replay from the durable 998-row E357 corpus. It raises
slot-component supervision from 1 to 4, adds focal gamma 2, and exposes the
honest slot contract in prompt context. The CPU HF-context run stopped normally
at its 5,000-target-token budget after 101 steps and 79.6 seconds, within both
the three-minute harness budget and external 170-second process cap.

The checkpoint and full state were uploaded and verified at
`hf://buckets/TKendrick/OpenUI/checkpoints/e513-e396-e500-replay050-slotrole4-focal2-r3-5k`.
The serving checkpoint SHA-256 is
`59253c679477060694370c5e2d8cd9fce5d7accc7d71df3b6d56edf0a88a9548`.

E514 evaluates that exact checkpoint on the four-record OOD suite under E510's
leading component-plan weight-4 policy. It emits AgentEvals JSONL and a pinned
AgentV bundle without execution errors.

| Checkpoint | n | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | Strict v2 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E505 / E510 | 4 | 0.50 | 0.6583 | 0.3446 | 0.3958 | 0.8405 | 0.4679 | 0.1625 | 0.00 | 0/1 |
| E513 / E514 | 4 | 0.00 | 0.4917 | 0.2750 | 0.2083 | 0.7695 | 0.3500 | 0.0625 | 0.00 | 0/1 |

## Decision

Reject E513 for promotion and do not expand it to the other suites. The
training run and bucket checkpoint remain durable diagnostic evidence, but the
stronger slot-role objective does not preserve the component-plan gain. The
next iteration should target role-labeled examples or isolate the focal-loss
effect with a matched control; increasing this loss further is unsupported.

Exact recipe, provenance, metrics, and gate outcome:
[machine-readable record](iter-e513-slot-role-supervision-20260719.json).
