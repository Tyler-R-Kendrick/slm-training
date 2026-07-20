# E605 — missing-family score-scale trace

Date: 2026-07-20
Status: completed diagnostic; not promotable or ship

E605 instruments every post-first top-level semantic-plan decision without
changing decode scores. Both capped OOD `n=4` runs use the E604 weight-32
policy and checkpoint. The first run completed normally but retained only the
top eight post-plan candidates, which hid the deeply negative planned `Card`
score. It is retained as an instrumentation-negative result. The corrected
retry always records every positively biased planned candidate.

Both runs exactly reproduce E604: syntax 1.0, meaningful-v1 0.5, strict
meaning-v2 0, fidelity 0.5917, validity 0.7550, structure 0.5756, component
recall 0.6250, reward 0.8145, AST-node F1 0.6111, AST-edge F1 0.4643, and
AgentV 0/1.

The corrected trace localizes the dashboard failure:

- `Callout` receives +32 at positions 4 and 7. It initially remains 44.28
  points below `TextContent`, then wins at position 7 and is emitted.
- `Card` is legal and receives +32 at all 16 observed post-first choices. Its
  learned base score remains −33.84 to −37.79; after bias it is still
  −1.84 to −5.79 and loses by 30.40 to 44.28 points.
- `Modal`, the second `Input`, and `Button` receive +32 from competitive base
  scores and are emitted. No later score layer changes a plan-stage winner.

This rejects candidate pruning, token-identity mismatch, and downstream
overwrites as the cause. It also explains why weights 16 and 32 were
ineffective: raw additive plan scores are not calibrated to the model's
family-specific legal-token logits. Do not continue an arbitrary global
weight sweep. The next bounded lever should normalize planned-family scores
against the best legal component or enforce an explicit finite margin, with
duplicate/cardinality safeguards and the same honest acceptance gates.

Neither run created, promoted, or synced a checkpoint.

Evidence: [JSON](iter-e605-missing-family-score-trace-20260720.json).
