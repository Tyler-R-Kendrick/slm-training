# E508 — default-generation length-safe OOD replication

E508 tests the leading E507 policy at the checkpoint's default eight generation
steps and four decode attempts. It uses the same rejected E505 checkpoint, all
four OOD records, constrained slot-contract grammar-LTR decode, and a
length-safe 160-token canvas.

The evaluation completed under its external 170-second cap and emitted
AgentEvals plus pinned AgentV evidence without execution errors.

| n | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | p50 latency | AgentV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.0 | 0.25 | 0.2583 | 0.2281 | 0.3333 | 0.692 | 0.3389 | 6,803 ms | 0/1 |

Every quality metric exactly matches E507's four-step, one-attempt result.
Grammar-LTR primary decode succeeds on its first path, so denoising-step and
retry controls do not affect this comparison.

## Decision

Keep constrained slot-contract grammar-LTR decode as the leading diagnostic
policy. Stop spending experiments on denoising-step or retry settings for this
path. AgentV remains red; semantic component correctness is the remaining
blocker. No checkpoint was created or synced.

Exact metrics:
[machine-readable record](iter-e508-default-generation-ood-contract-decode-20260719.json).
