# E611 — repeated-plan distinct-slot allocation

Date: 2026-07-20
Status: completed, retained as the scratch baseline, not promotable

E611 adds a default-off, compiler-local margin for repeated prompt-plan
families. While the active repeated instance has not consumed a visible slot,
the highest-scoring legal slot that has not appeared in an earlier instance is
floored two points above the current legal maximum. The lever uses only the
public slot contract, prompt plan, legal candidates, and generated prefix.

The matched OOD `n=4` replay completed normally. Dashboard now assigns
`:ood.dash.m1.value` and `:ood.dash.m2.value` to different Cards, shortens from
60 to 54 output symbols, and improves fidelity from 0.60 to 0.80, validity from
0.76 to 0.88, structure from 0.7417 to 0.7750, and reward from 0.865 to 0.925.
Modal and auth remain prediction-identical; gallery remains an empty
`ImageGallery`.

Aggregate meaningful-v1 remains 0.75 while fidelity improves 0.65→0.70,
validity 0.69→0.72, structure 0.7646→0.7729, reward 0.6998→0.7148, AST-node F1
0.7437→0.7579, AST-edge F1 0.6181→0.6310, and p95 latency falls
12.98→11.46 seconds. The preregistered no-regression condition passes. Retain
the lever as the next scratch baseline.

Strict meaning-v2 remains zero and AgentV remains 0/1, so this is not a
promotion or ship result. The next iteration should prevent an authored
collection component from closing its typed child array empty while visible
slots remain, then measure whether the legal child-family choice repairs the
gallery case.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e611-repeated-plan-slot-allocation-20260720.json).
