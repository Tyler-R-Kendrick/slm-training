# Autotrain c1803: prefix-LTR scaffold supervision is null

**Verdict:** reject `ltr_prefix_loss_weight=1` at this recipe. Both
size-matched arms completed 20 CPU scratch steps and all 3 smoke records.

| Arm | Params | Loss | Structure | Binder F1 | Recall | Fidelity | p50 ms | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| weight 0 control | 1,608,962 | 13.55472 | 0.419733 | 0.82222 | 0.16667 | 0.72222 | 1,827.59 | retain baseline only |
| prefix weight 1 | 1,608,962 | 13.47166 | 0.419733 | 0.82222 | 0.16667 | 0.72222 | 2,001.80 | reject approach |

Both arms also tie parse 1.0, meaningful-program rate .33333, and reward
.85367, with zero decode timeouts and complete AgentV bundles. The candidate's
.08307 lower final batch loss produces no quality gain and worsens p50 by
174.21 ms. Fixture ship gates fail.

The local explicit-no-sync checkpoint SHA-256 values are `74f1e6e1...e97ba`
(control) and `8dfb8866...fee0` (candidate). Neither is reusable, promotable,
synced, or ship evidence. Lean is `not_applicable:screening`.

Campaign harness v104 adds the next distinct zero-parameter arm:
`component_token_loss_weight=1` versus zero. It directly reweights masked
component-type output tokens, targeting the observed .16667 component recall,
and emits raw component/prefix/non-component token-loss attribution on every
training step. It changes neither grammar authority, decode legality, nor
capacity. If component recall remains null, use the emitted family CE and count
signals to decide whether the blocker is exposure, optimization, or constrained
choice ranking before introducing another objective.

Machine evidence:
[`autotrain-cycle-1803-scaffold-prefix-null.json`](autotrain-cycle-1803-scaffold-prefix-null.json).
