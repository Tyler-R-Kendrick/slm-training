# Autotrain c1858: slot-contract context replay null

**Verdict:** exact frozen replay completed; quality null; no ship claim.

After c1857's inventory-extraction repair, the control and
`slot_contract_in_context` candidate both completed the frozen smoke replay.
The candidate is size-matched and preserves every guarded quality metric, but
all quality remains below ship gates and exact AST/canonical agreement is zero.
The lower fixture p50 is an efficiency diagnostic only; it cannot promote a
quality-null arm.

| Arm | Params | Loss | Struct | MPR | Recall | Binder F1 | Fidelity | p50 ms | Tokens | Forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 7.9206 | .1742 | .333 | .25 | .633 | .528 | 942 | 21 | 4 |
| slot-contract-context | 1,608,962 | 7.9296 | .1742 | .333 | .25 | .633 | .528 | 871 | 21 | 4 |

The candidate is `7.66%` faster at this tiny fixture, with identical quality,
tokens, forwards, parameters, and zero exact matches. Smoke `n=3` is below the
required `n≥20`; held-out, adversarial, OOD, and RICO suites are absent. The
slot-contract arm is therefore exhausted as a quality approach and must not be
recycled or promoted. The repaired extractor and its c1857 harness failure
remain documented separately.

Machine evidence:
[`autotrain-cycle-1858-slot-contract-context-replay-null.json`](autotrain-cycle-1858-slot-contract-context-replay-null.json).
