# Autotrain c1763: finalization succeeds and exposes literal-close starvation

**Verdict:** the c1762 parent-finalization repair worked. The exact frozen
batch-size-one replay reused both c1760 checkpoints, completed both suite
evaluations, wrote gates, AgentEvals assertions, and AgentV bundles, and
returned a terminal failed outcome instead of exceeding the outer stage wall.
No model was retrained and no checkpoint was created. The candidate remains
unscoreable because all eight records reached typed internal decode timeouts.

| Arm | Train work | Params | Smoke | Held-out | AgentV | Decision |
| --- | --- | ---: | --- | --- | --- | --- |
| batch size 1 | cached c1760 checkpoint | 1,608,962 | n=3; 0 complete; 3 typed timeouts; p50 incl. incomplete 4,196.94 ms | n=5; 0 complete; 5 typed timeouts; p50 incl. incomplete 4,174.69 ms | 0/2 assertions; 0 execution errors | finalized model-decode failure; not scoreable |
| control, batch size 2 | cached c1760 checkpoint | 1,608,962 | n=3; parse 1; meaning .3333; structure .13527; recall .1667; p50 1,383.44 ms | n=5; parse 1; meaning 0; structure .06024; recall .02857; p50 1,358.75 ms | 0/2 assertions; 0 execution errors | complete gate rejection |

The candidate used a 37.0 s cumulative evaluation wall. Fair-shared effective
per-record caps were 4.1961-4.1967 s for smoke and 4.1710-4.1742 s for
held-out. The five-second parent finalization tail was sufficient: this cycle's
terminal state is an experiment failure caused by incomplete evaluation
measurements, not `stage exceeded wall-time limit`. The orchestration repair is
therefore verified at its intended boundary.

The constrained-selection traces identify a model-ranking failure rather than
an empty grammar domain. Every one of the eight candidate records selects
`Slider`, enters its numeric argument, and repeatedly selects legal byte tokens
such as `B:31` and `B:36`. Once inside the literal, each step still has 12 legal
candidates, but the model never ranks the legal terminator high enough. One
representative prefix grows from
`root = Slider("$6", "discrete", 111111` to an ever-longer digit string until
the per-record wall fires. Grammar legality remains enforced; no unconstrained
fallback occurred.

The next experiment is a new, size-matched `literal-close` arm, not another
replay of the stalled checkpoint. It exposes the already implemented
`ltr_tail_loss_weight` through the governed autoresearch schema and compares
weight 2.0 against the matched 0.0 control with identical parameter count,
steps, batch size, data, and constrained decoder. The hypothesis is deliberately
falsifiable: stronger suffix/termination supervision must reduce typed timeouts
without lowering parse rate or binder-reference F1. If it does not, the next
hypothesis should target a grammar-transition-aware termination margin rather
than changing timeouts or weakening I6.

Lean is `not_applicable:no_champion`: this fixture replay produced no promotion
candidate. The promotion-integrated Lean/formal lane remains mandatory when a
champion exists; it was not silently omitted.

Machine evidence:
[`autotrain-cycle-1763-numeric-literal-close-starvation.json`](autotrain-cycle-1763-numeric-literal-close-starvation.json).
