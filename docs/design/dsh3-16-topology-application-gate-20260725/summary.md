# DSH3-16 topology-application activation gate (SLM-384)

Date: 2026-07-25
Status: not authorized; closed in plan-only mode
Scope: activation-gate evaluation only; no model, checkpoint, or ship claim

## Decision

hierarchical_head verdict is 'rejected', not an accepted SUPPORTED operator-selection mechanism. DSH3-16's topology-apply comparison has no accepted arm to lower or apply, and no ActionEffectV1-to-TopologyNode mutation bridge exists between the operators package's AST-dict representation and the topology-diffusion model's typed node tree. Close in plan-only mode without production model code (SLM-250/LOT1-01 precedent).

## Gate result

- `verdict`: **not_authorized**
- `hierarchical_head_verdict`: `rejected`
- `hierarchical_head_evidence_ids`: SLM-383.hierarchical_operator_head_baseline

SLM-383's causal-attribution experiment (`hierarchical_operator_head_baseline`)
found the encoder-side hierarchical head strictly beats its own weight-zero
capacity control but ties the token baseline -- it does not causally improve
beyond it, so it was rejected rather than left unrun. DSH3-16's own design
compares recompute-only, hierarchical-head-with-ordinary-lowering, and
topology-apply arms; without an accepted operator-selection mechanism there
is no arm to lower or apply, and separately no `ActionEffectV1`-to-
`TopologyNode` mutation bridge exists between the operators package's
AST-dict representation and the topology-diffusion model's typed node tree
(`grammar_diffusion.TopologyNode`). Building that bridge and the node-pass/
model-forward instrumentation the issue calls for is out of scope while the
gate is closed.

AgentV passed 3/3 cases with mean 1.0 and 0 execution errors.

No checkpoint was created, so the model card and README checkpoint summary
do not change. This evaluation consumes the immutable DSH3-17/SLM-385
disposition report and does not rerun or reinterpret its evidence.
