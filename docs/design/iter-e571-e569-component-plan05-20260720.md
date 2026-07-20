# E571 — E569 component-plan decode weight 0.5

E571 reuses E569's checkpoint and evaluates its trained component-plan head at
decode weight 0.5 under E570's matched LTR policy. The head applies 107 times
but changes only one choice.

On matched OOD `n=4`, every quality aggregate equals E569: fidelity 0.2583,
structure 0.2031, component recall 0.3333, reward 0.6920, AST-node F1 0.3389,
AST-edge F1 0, and meaningful-v1 0.25. Binding-aware meaningful-v2 remains 0,
and AgentV remains 0/1. Relative to E570 weight 1, weight 0.5 recovers recall
and meaningful-v1 but gives back all topology/reward gains. No checkpoint was
created.

An earlier completed control (`e571-e569-component-plan05-eval-r1`) inherited
E569's `grammar_ltr_primary=false` instead of E570's `true`; it produced the
same aggregates and is retained as a disclosed recipe-mismatch control.
Pre-evaluation failures for a missing test path and omitted honest-role context
produced no scored evidence.

**Verdict:** retain weight 0.5 only as the E569-equivalent coverage setting.
The 0.5-to-1.0 ladder is a sharp threshold rather than a smooth Pareto; stop
this scalar ladder and return to training or a targeted strict-semantic
mechanism. Evidence:
[JSON](iter-e571-e569-component-plan05-20260720.json).
