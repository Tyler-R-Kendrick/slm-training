# E553 — corpus-local proportional slot priors

E553 replaces symmetric slot-owner pseudo-counts with class-prior-proportional
smoothing and makes warm-start training rebuild deterministic priors from the
active corpus after restoring learned weights. This prevents an unseen
token/component pair from receiving a positive association merely because its
component class is rare.

Two attempts were excluded before interpretation. R1 restored the stale parent
prior and therefore never exercised the treatment. R2 accidentally enabled
DESIGN.md context and was not matched to E547. The valid R3 run used the exact
E547 honesty recipe, processed 1,304 target tokens in 34.48 seconds under
`max_wall_minutes=3`, and wrote local checkpoint SHA
`510e55cf16fe23edd4ac408ed37d2409a895143646a6321c5b491c148e75399d`.

The rebuilt prior changes all 3,420 scores comparable with the parent table,
reduces positive scores from 2,381 to 100, and flips 2,303 parent-positive
scores negative. The matched OOD `n=4` result reaches fidelity 0.3000, validity
0.4800, structure 0.1244, component recall 0.0625, reward 0.5453, and AST node
F1 0.1556. Meaningful-v1, strict-v2, AST edge F1, and AgentV remain zero.

Relative to E547, fidelity improves 0.0417 and reward 0.0050, but structure
falls 0.1004, recall falls 0.1458, and AST node F1 falls 0.1714.

**Verdict:** keep the generalized prior and warm-start correctness fixes, but
reject the checkpoint. Prior calibration is exhausted on this subset; the next
lever should change corpus or supervision coverage. Evidence:
[JSON](iter-e553-slot-prior-proportional-smoothing-20260720.json).
