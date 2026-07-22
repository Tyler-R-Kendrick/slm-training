# E724 symbol-only slot-coverage closure

**Date:** 2026-07-21  
**Decision:** reject as no-effect; retain E723 unchanged  
**Evidence:** [`iter-e724-slot-coverage-close-symbol-only-20260721.json`](iter-e724-slot-coverage-close-symbol-only-20260721.json)

E723's valid outputs close after one slot-owning child, leaving requested
components missing. E724 therefore applies the existing minimal effective
`slot_coverage_close_decode_weight=2` to the exact E723 checkpoint and retained
slot-owner recipe. This is a local CPU, honest-contract smoke diagnostic (`n=3`),
one attempt, 160-symbol canvas, eight-second per-record timeout, and no fallback.
No training ran and no checkpoint was created.

E724 is prediction-identical to E723: prediction-set SHA
`cc795b50ff56b342909c033a6b88735bda67f0e293ea4704f500b006016156f0`.
Parse is 1.0, meaning-v1 0.6667, strict-v2 0.0, fidelity 0.5278, structure
0.5614, recall 0.4167, reward 0.8073, and AgentV 0/1. No coverage-close
application or choice-change counter is emitted.

The policy cannot prove a compatible continuation at this root binding/list
termination path, so increasing its weight cannot help. Reject E724 and close
the coverage-close weight ladder for this checkpoint. The next arm should add
the learned prompt component-inventory signal to E723 rather than modify
closure scoring.
