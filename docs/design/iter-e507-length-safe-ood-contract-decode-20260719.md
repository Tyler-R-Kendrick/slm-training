# E507 — length-safe OOD contract-decode replication

E507 removes E506's known canvas-length confound on the most discriminating
suite. OOD gold p95 is 143 tokens; both matched evaluations use a 160-token
canvas and the same rejected E505 checkpoint.

## Matched recipe

Both arms use all four OOD records, CPU/frozen local SmolLM2-135M context,
honest slot-contract scoring, grammar-constrained LTR decode, no DESIGN
context, no fallback, four generation steps, and one attempt. They differ only
in constrained slot-contract decode. Each process completed under its external
170-second cap and emitted AgentEvals plus pinned AgentV evidence.

| Decode | n | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | p50 latency | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contract off | 4 | 1.0 | 0.0 | 0.0 | 0.0729 | 0.0 | 0.0 | 0.0313 | 4,571 ms | 0/1 |
| Contract on | 4 | 1.0 | **0.25** | **0.2583** | **0.2281** | **0.3333** | **0.692** | **0.3389** | 6,966 ms | 0/1 |

Every quality metric in both arms is identical to E506's 96-token OOD result.
The constrained decode gain is therefore not a truncation artifact. The
length-safe policy costs about 2.40 seconds at p50 in this small CPU run.

## Decision

Retain constrained slot-contract decode as the leading inference policy.
Length safety is verified on OOD, but promotion remains blocked: AgentV is
still 0/1 and the diagnostic uses only four generation steps and one attempt.
No checkpoint was created or synced.

Exact metrics:
[machine-readable record](iter-e507-length-safe-ood-contract-decode-20260719.json).
