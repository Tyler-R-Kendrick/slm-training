# Autotrain c1851: slot-component coverage

**Verdict:** real fixture learning signal, rejected for ship and promotion.

The implemented slot-component owner trained successfully and improved
meaningful-program rate from `0` to `.667`, while holding binder F1 and fidelity
at `1.0`. It also improved structural similarity only `.1425 -> .1767` and
raised training loss, latency (`7400 -> 8771 ms`), tokens (`184 -> 214`),
forwards (`39 -> 44`), compiler time (`19515 -> 23354 ms`), and parameters
(`1,608,962 -> 1,613,482`). The smoke suite has only three documents and exact
AST/canonical agreement remains zero.

| Arm | Params | Loss | Struct | MPR | Binder F1 | Fidelity | Reward | p50 ms | Tokens | Forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware control | 1,608,962 | 20.9509 | .1425 | 0 | 1.0 | 1.0 | .981 | 7400 | 184 | 39 |
| slot-component coverage | 1,613,482 | 22.5437 | .1767 | .667 | 1.0 | 1.0 | .989 | 8771 | 214 | 44 |

This is evidence the model can learn a targeted component signal, but it is not
evidence of high-quality production OpenUI generation. Parameter growth is
charged, the latency tradeoff is outside the bounded screening budget, and
held-out suites are absent. Keep the matched control and require a new
size-matched objective before another model hypothesis.

Machine evidence:
[`autotrain-cycle-1851-slot-component-coverage-null.json`](autotrain-cycle-1851-slot-component-coverage-null.json).
