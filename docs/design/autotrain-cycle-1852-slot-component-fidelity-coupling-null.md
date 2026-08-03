# Autotrain c1852: slot-component/fidelity coupling

**Verdict:** null fixture result, rejected for promotion and ship.

The distinct coupling trained successfully, but every guarded quality metric
matched the control exactly. It increased loss, parameters, decode work, and
latency, so it provides no evidence that the added fidelity objective helps
the implemented slot-component owner.

| Arm | Params | Loss | Struct | MPR | Recall | Binder F1 | Fidelity | p50 ms | Tokens | Forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware control | 1,608,962 | 12.0027 | .1742 | .333 | .25 | .633 | .528 | 910 | 21 | 4 |
| slot-component + fidelity | 1,613,477 | 24.1579 | .1742 | .333 | .25 | .633 | .528 | 966 | 30 | 5 |

Smoke `n=3`, exact AST/canonical agreement is `0`, and held-out,
adversarial, OOD, and `rico_held` suites are absent. The candidate is therefore
not learning a measurable new capability in this comparison; the remaining
blockers are objective signal strength, evaluation volume, and parameter/
latency cost. Keep the matched control and require a new preregistered,
size-matched objective before another screening run.

Machine evidence:
[`autotrain-cycle-1852-slot-component-fidelity-coupling-null.json`](autotrain-cycle-1852-slot-component-fidelity-coupling-null.json).
