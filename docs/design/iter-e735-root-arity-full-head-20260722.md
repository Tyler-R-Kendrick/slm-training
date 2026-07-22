# E735 full-head root-arity supervision

**Date:** 2026-07-22  
**Decision:** retain the objective fix; reject the checkpoint  
**Evidence:** [`iter-e735-root-arity-full-head-20260722.json`](iter-e735-root-arity-full-head-20260722.json)

E735 fixes the mismatch exposed by E734. Root-reference arity training formerly
sliced each row's classifier to its locally feasible classes, while inference
ranked the full head. The unused tail therefore received no negative gradient:
E731 predicted impossible class 41 on all three smoke prompts, explaining why
stronger decoder authority over-continued until timeout.

The canonical TwoTower objective now computes cross-entropy and accuracy over
the full inference head. Tests seed an impossible tail class with dominant
logit and prove that it is counted as an error and receives a suppressing
gradient for both choice and lexer tokenizers.

The matched local CPU train uses the same 141-record symbol-only corpus and
E731 recipe. It completes 140 steps in 82.07 seconds under
`max_wall_minutes=2`. The corrected accuracy is stricter and not directly
comparable to E731's masked metric (mean 0.2393 versus masked 0.3548), while
reconstruction and slot-owner metrics remain identical. On the same three
visible smoke prompts, the E735 head predicts class 1 with top classes
1/2/3/0/4 instead of E731's class 41.

Matched strict compiler-tree decode weights 0 and 1 are prediction- and
metric-identical: parse 1.0, meaning-v1 0.6667, strict-v2 0.0, fidelity 0.5278,
structure 0.5614, recall 0.4167, reward 0.8073, and AgentV 0/1. Weight 1 applies
six times, changes zero choices, and has zero timeouts. Retain the generalized
training correction because it aligns training with inference and removes the
untrained tail. Reject, do not sync, and do not promote this scratch checkpoint
because it adds no quality gain.

The initial corpus-audit command failed before emitting evidence because it
queried nonexistent `source_family`; the corrected audit used `source` and
found 66/141 eligible records with arity targets 0:5, 1:28, 2:15, 3:14, 4:4.
This ruled out adding another sampling knob as the first response.
