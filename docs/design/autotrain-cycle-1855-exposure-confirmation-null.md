# Autotrain c1855: exposure-cap fresh confirmation

**Verdict:** fresh-seed confirmation rejected; no champion or ship claim.

The candidate improved the fixture primary again, but the confirmation did not
re-establish the complete quality contract. MPR and component recall stayed
below gate floors, exact AST/canonical agreement stayed zero, and decode work
and latency increased. The campaign therefore exhausts this fingerprint rather
than spending more scalar steps.

| Arm | Params | Loss | Struct | MPR | Recall | Binder F1 | Fidelity | Reward | p50 ms | Tokens | Forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh control | 1,608,962 | 10.2016 | .230 | 0 | 0 | 0 | 0 | 0 | 2433 | 60 | 7 |
| fresh exposure cap | 1,613,477 | 18.4581 | .354 | .333 | .333 | .633 | .528 | .807 | 2574 | 72 | 12 |

The primary rose `.1242`, but the candidate is `5.81%` slower, uses `20%`
more tokens and `71%` more forwards, has no exact matches, and remains smoke
`n=3`. Training loss also diverged from certified quality (`10.20 -> 18.46`),
so loss is diagnostic only. Lean promotion preflight remains locked; a distinct
quality-targeted objective is required.

Machine evidence:
[`autotrain-cycle-1855-exposure-confirmation-null.json`](autotrain-cycle-1855-exposure-confirmation-null.json).
