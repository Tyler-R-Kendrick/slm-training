# E506 — larger constrained slot-contract decode diagnostic

E506 tests E505's three-record decode signal on every committed held-out, OOD,
and adversarial record: 13 examples total. Both arms use the same rejected E505
checkpoint and differ only in constrained slot-contract decode.

## Matched recipe

Both evaluations use CPU, frozen local SmolLM2-135M context, honest
slot-contract scoring, grammar-constrained LTR decode, no DESIGN context, no
fallback, four generation steps, one attempt, and a 96-token canvas. Each
process was externally capped at 170 seconds and completed within the cap.
Every suite emitted AgentEvals and a pinned AgentV result without execution
errors.

The 96-token canvas is below gold p95 for held-out (160), OOD (143), and
adversarial (110). These results are diagnostic and cannot support a ship claim.

| Decode | n | Syntax | Meaningful | Fidelity | Structure | Recall | Reward | AST node F1 | AST edge F1 | AgentV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contract off | 13 | 1.0 | 0.0 | 0.0 | 0.1271 | 0.1410 | 0.0 | 0.1524 | 0.0192 | 0/3 |
| Contract on | 13 | 1.0 | **0.1538** | **0.2538** | **0.1669** | **0.2654** | **0.5454** | **0.2385** | 0.0 | 0/3 |

Constrained slot-contract decode improves meaningful rate by 0.1538, fidelity
by 0.2538, structure by 0.0399, recall by 0.1244, reward by 0.5454, and AST
node F1 by 0.0862. It loses 0.0192 AST edge F1. OOD and adversarial each reach
meaningful rate 0.25; held-out remains zero.

## Persistence and decision

The control-plane reader now preserves native multi-suite matched results
instead of forcing every committed experiment arm into a synthetic smoke row.
No checkpoint was created or promoted.

Keep constrained slot-contract decode as the leading E505 inference policy,
but do not claim ship readiness. Next rerun the most discriminating suite with
a length-safe canvas under the same process cap.

Exact metrics:
[machine-readable record](iter-e506-slot-contract-decode-20260719.json).
