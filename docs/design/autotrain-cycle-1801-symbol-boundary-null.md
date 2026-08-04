# Autotrain c1801: symbol-boundary supervision is null

**Verdict:** reject `symbol_boundary_loss_weight=1` at this recipe. Both
size-matched arms completed 21 CPU scratch steps and all 3 smoke records.

| Arm | Params | Loss | Structure | Binder F1 | Fidelity | p50 ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| weight 0 control | 1,608,962 | 12.48600 | 0.135267 | 0.63333 | 0.52778 | 938.53 | retain baseline only |
| boundary weight 1 | 1,608,962 | 21.78877 | 0.135267 | 0.63333 | 0.52778 | 949.90 | reject approach |

Both arms also tie parse 1.0, meaningful-program rate 0.33333, component recall
0.16667, and reward 0.76533, with zero decode timeouts and complete AgentV
bundles. The candidate has exactly zero primary/binder/fidelity gain, 9.30276
higher loss, and 11.37 ms worse p50. Fixture ship gates fail.

The local explicit-no-sync checkpoint SHA-256 values are `d8b45266...df831`
(control) and `34783017...fd21a` (candidate). Neither is reusable, promotable,
synced, or ship evidence. Lean is `not_applicable:screening`.

Campaign harness v102 adds the next distinct zero-parameter arm:
`design_md_dropout=0.25` versus zero. Deterministic record-level context dropout
tests whether the fixture model is copying DESIGN.md scaffolds instead of
learning prompt-to-grammar structure; it does not change grammar authority,
decode legality, or capacity. If that arm is null, prioritize coverage and
per-family loss instrumentation before adding another local loss.

Machine evidence:
[`autotrain-cycle-1801-symbol-boundary-null.json`](autotrain-cycle-1801-symbol-boundary-null.json).
