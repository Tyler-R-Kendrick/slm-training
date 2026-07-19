# E542 — learned root-reference arity

E542 replaces the rejected hand-written reference-completeness heuristic with
an isolated choice-codec head that predicts the number of direct generated
references in the terminal structural root list. The target is
dependency-order aware: nested lists are ignored and the final completed list
is supervised. It covers 188/244 E530 records (77.05%); observed targets are
1:34, 2:102, 3:47, and 4:5.

The local-only scratch continuation
`e542-e531-root-reference-arity1-r1-24s` initialized from the bucket-verified
E531 checkpoint and completed all 24 CPU steps in 52.93 seconds under
`max_wall_minutes=3`. It saw 1,270 target tokens. Auxiliary loss moved from
3.9565 on step 1 to 3.3124 on step 24, with late tiny-batch accuracy becoming
nonzero. The serving checkpoint SHA is
`2d5cd4b3c8c721e8193e06b5aa231bd9ec5009b4bec9cacfeebe842f6854c5d8`.
This was an explicit `--no-sync-checkpoints` scratch diagnostic, not a full
HF-context train or promotion.

The matched four-record OOD control with root-arity decode disabled reached
syntax 1.0, meaningful-v1 0.50, fidelity 0.5917, validity 0.7550, structure
0.3019, component recall 0.4167, reward 0.7950, AST node F1 0.3271, and AST
edge F1 0.0333. Strict binding-aware meaning remained 0.0 and AgentV failed
0/1.

The first weight-1 replay exposed an implementation defect rather than a model
gain: 32 applications changed 13 choices but impossible unobserved tokenizer
tail classes collectively forced one root to 31 references. Its quality
metrics exactly matched control. The decoder was then bounded by the number of
available generated sections; a regression test proves that even a dominant
impossible tail logit is ignored.

The clean bounded replay reduced the intervention to 11 applications and 7
choice changes. Every quality metric again exactly matched the weight-zero
control, strict meaning remained 0.0, and AgentV remained 0/1.

**Verdict:** retain the generalized target, isolated auxiliary head, semantic
bound, and telemetry. Keep root-reference arity decode default-off and reject
weight 1 for promotion. The short continuation checkpoint is useful diagnostic
evidence but is not a ship checkpoint; the next experiment should improve
head calibration or root-reference identity supervision and must use a matched
control. Machine-readable evidence:
[JSON](iter-e542-learned-root-reference-arity-20260719.json).
