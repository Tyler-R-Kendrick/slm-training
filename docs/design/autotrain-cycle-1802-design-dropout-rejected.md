# Autotrain c1802: DESIGN context dropout regresses structure

**Verdict:** reject `design_md_dropout=0.25` at this recipe. Both size-matched
arms completed 22 CPU scratch steps and all 3 smoke records.

| Arm | Params | Loss | Structure | Binder F1 | Recall | MPR | p50 ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dropout 0 control | 1,608,962 | 9.77485 | 0.174167 | 0.63333 | 0.25000 | 0.33333 | 946.76 | retain baseline only |
| dropout .25 | 1,608,962 | 10.03550 | 0.096400 | 0.63333 | 0.08333 | 0 | 944.05 | reject approach |

Both arms retain parse 1.0, fidelity .52778, and reward .76533, with zero
decode timeouts and complete AgentV bundles. The candidate loses .077767
structure, .16667 component recall, and .33333 meaningful-program rate. Its
2.71 ms p50 improvement cannot override those quality regressions. Fixture ship
gates fail.

The local explicit-no-sync checkpoint SHA-256 values are `dd9d11e8...7154`
(control) and `212ba7ce...573b` (candidate). Neither is reusable, promotable,
synced, or ship evidence. Lean is `not_applicable:screening`.

Campaign harness v103 adds the next distinct zero-parameter arm:
`ltr_prefix_loss_weight=1` versus zero. Prefix supervision directly targets
early scaffold formation after context dropout exposed reliance on copied
scaffolds; it changes neither grammar authority, decode legality, nor capacity.
If it is null, prioritize token-position and component-family loss attribution
before adding another local auxiliary objective.

Machine evidence:
[`autotrain-cycle-1802-design-dropout-rejected.json`](autotrain-cycle-1802-design-dropout-rejected.json).
