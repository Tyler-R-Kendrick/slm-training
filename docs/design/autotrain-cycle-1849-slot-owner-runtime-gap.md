# Autotrain c1849: slot-owner runtime gap

**Verdict:** incomplete infrastructure measurement, not a model result.

The corrected tree-mode arm reached the model constructor, which then rejected
the reserved binder-slot ownership fields because no runtime owner is
implemented. The matched control completed; the candidate did not train or
evaluate. This is why the supervisor keeps the frozen replay action pending.

| Arm | Status | Loss | Structure | MPR | Binder F1 | Fidelity | p50 ms | Checkpoint |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| capacity-aware control | complete, fixture | 17.5136 | .0575 | 0 | .8222 | .7222 | 1059.1 | `44db58f1...8f00` |
| binder-slot-ownership | runtime owner missing | — | — | — | — | — | — | none |

Commit `fb9093314` replaces this unusable reserved arm with the implemented
`slot-component-coverage` owner, preserving the model's fail-closed guard. The
next cycle must use that distinct owner; c1849 itself is never a model null or
ship evidence.

Machine evidence:
[`autotrain-cycle-1849-slot-owner-runtime-gap.json`](autotrain-cycle-1849-slot-owner-runtime-gap.json).
