# Autotrain c1733: component-edge screen

**Verdict:** the exactly size-matched component-edge arm is a complete quality
null. It matches the control on every quality metric and deterministic-work
counter. Its 2.13% fixture p50 improvement is below policy v4's 5% efficiency
floor, so the arm is rejected and is not checkpoint, promotion, or ship evidence.

## Result matrix

| Arm | Params | n | Parse | Binder F1 | Meaningful | Structure | Recall | AST node / edge F1 | p50 | Disposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,766,987 | 3 | 1.0 | 1.0 | 0 | .20693 | .08333 | .05128 / 0 | 6,136.43 ms | complete fixture control; gates fail |
| component-edge loss `1.0` | 1,766,987 | 3 | 1.0 | 1.0 | 0 | .20693 | .08333 | .05128 / 0 | 6,005.56 ms | exact quality null; 2.13% below floor |

Both arms also match placeholder fidelity 1.0, reward 0, 82 neural forwards,
88,203 unique completion states, 7,330 witness expansions, and 95,213 parser
forks. Candidate training loss is worse (9.0292 vs 8.0219); training wall is
3.288 vs 3.450 seconds. These are 24-step CPU scratch checkpoints, seed 101733,
batch 2, with the component-edge head prebuilt in both arms.

## Honest gate and formal state

AgentV completed both arms with zero execution errors. Smoke `n=3` is below the
evidence floor and both arms miss meaningful, structural, component-recall, AST
BEq, canonical BEq, and reward gates. Held-out, adversarial, OOD, and
`rico_held` were not run. Lean is `not_applicable:screening`; no champion exists
and no formal promotion claim was attempted.

## Harness signal and repair

The raw c1733 handoff correctly exhausted component-edge but proposed
component-plan next, even though c1731 and c1732 had already completed that
family as non-positive. The successor selector only cooled down open champion
queue entries and the immediately completed candidate; it did not consume
complete nulls across the predecessor chain.

The repair collects complete non-positive arm families over one bounded pass of
the screening bank, combines them with open-queue exclusions, and chooses the
first unexhausted quality alternative. On the c1733 lineage the exhausted set is
`binder-topology`, `component-plan`, and `component-edge`; the repaired next
hypothesis is the exactly size-matched `component-inventory` arm. The cooldown
is bounded, so a family becomes eligible again after a full bank pass or a new
preregistered path instead of being silently banned forever.

## Next-run priorities

1. Run the size-matched component-inventory hypothesis; do not replay unchanged
   component-plan or component-edge supervision.
2. Preserve the matched structural-head capacity, smoke primary, exact work
   counters, and 5% efficiency floor.
3. Keep RL and promotion locked: this fixture has zero meaningful programs and
   no held-out or LeverProof evidence.
4. If component-inventory is another exact null, use its signal to rotate to the
   next unexhausted causal family rather than revisiting a recent null.

Machine-readable evidence is in
[`autotrain-cycle-1733-component-edge-screen.json`](autotrain-cycle-1733-component-edge-screen.json).
