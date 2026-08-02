# Autotrain c1788: binder-arity null

**Verdict:** reject. The exact frozen replay completed after the c1787 CLI
repair, but the size-matched binder-arity objective does not change any smoke
quality metric. Its 3.72% p50 improvement is below the 5% efficiency screen,
meaningful-program rate and binder-reference F1 are both zero, and training
loss is 30.4% worse than control.

| Arm | Params / train | Smoke | Decision |
| --- | --- | --- | --- |
| binder-arity | 2,145,602; 22 steps; loss 15.82408; 3.01 s | n=3; parse 1; meaning 0; structure .20583; binder F1 0; recall .0833; fidelity 0; reward 0; p50 2,652.77 ms | reject: exact quality null; p50 -3.72% only |
| matched control | 2,145,602; 22 steps; loss 12.13127; 3.39 s | n=3; parse 1; meaning 0; structure .20583; binder F1 0; recall .0833; fidelity 0; reward 0; p50 2,755.21 ms | baseline |

Both AgentV bundles completed with zero execution errors and both arms fail
ship gates. The run is a three-record CPU fixture screen, not generalization or
ship evidence. Both explicit no-sync checkpoints are provenance-only and must
not be reused, promoted, synced, or shipped.

The null closes this binder-arity approach at the current recipe; it does not
waive the binder-quality goal. The ranked next hypothesis is the distinct,
size-matched `binder-component-plan` train-and-decode objective. That arm should
prioritize binder-reference F1 and certified structural similarity while
holding parse, parameter count, and the latency budget.

Lean is `not_applicable:screening`: there is no confirmed champion and no formal
promotion target. The c1787 repair itself passed the full Lean build and axiom
audit locally and in CI.

Machine evidence:
[`autotrain-cycle-1788-binder-arity-null.json`](autotrain-cycle-1788-binder-arity-null.json).
