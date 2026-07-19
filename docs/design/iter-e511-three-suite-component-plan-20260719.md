# E511 — length-safe three-suite component-plan decode

E511 expands E510's leading component-plan weight-4 policy to every held-out,
OOD, and adversarial record. The 192-token canvas exceeds each suite's gold p95
(160, 143, and 110), while preserving honest slot-contract context and
constraint, default generation controls, no DESIGN context, and no fallback.

The evaluation completed under its external 170-second cap and emitted
AgentEvals plus pinned AgentV evidence without execution errors.

| Suite | n | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | Strict v2 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Held-out | 5 | 0.20 | 0.6200 | 0.2729 | 0.3167 | 0.6296 | 0.3600 | 0.1000 | 0.0 | 0/1 |
| OOD | 4 | 0.50 | 0.6583 | 0.3446 | 0.3958 | 0.8405 | 0.4679 | 0.1625 | 0.0 | 0/1 |
| Adversarial | 4 | 0.50 | 0.7500 | 0.4324 | 0.7083 | 0.4110 | 0.5947 | 0.2806 | 0.0 | 0/1 |
| Aggregate | 13 | 0.3846 | 0.6718 | 0.3440 | 0.4615 | 0.6272 | 0.4654 | 0.1748 | 0.0 | 0/3 |

OOD exactly reproduces E510 quality despite the larger canvas. Compared with
the earlier E506 three-suite policy bundle, every aggregate quality metric is
higher, but E506 is not an isolated causal control because its canvas,
generation settings, and context policy differ.

## Decision

Retain component-plan weight 4 as the leading diagnostic inference policy.
The remaining cross-suite failures are required-placeholder omissions,
placeholder semantic-role mismatches, placeholder spam, and schema value-role
mismatches. Target those behaviors next. AgentV remains red; no checkpoint was
created, promoted, or synced.

Exact metrics:
[machine-readable record](iter-e511-three-suite-component-plan-20260719.json).
