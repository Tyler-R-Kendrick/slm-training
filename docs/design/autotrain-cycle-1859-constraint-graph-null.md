# Autotrain c1859: constraint-graph null

**Verdict:** size-matched fixture null; no ship claim.

Grammar constraint-graph mode completed cleanly under the production
grammar-constrained path, but it produced exactly the same loss, quality,
tokens, and forwards as its matched control. Its p50 was 3.44% slower. Smoke
`n=3` is below the required `n≥20`, all exact metrics are zero, and production
suites are absent. The arm is exhausted as a quality approach; the next run
needs a genuinely new preregistered objective.

| Arm | Params | Loss | Struct | MPR | Recall | Binder F1 | Fidelity | p50 ms | Tokens | Forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | 12.4062 | .3058 | .333 | .25 | .952 | .917 | 2536 | 87 | 18 |
| constraint-graph | 1,608,962 | 12.4062 | .3058 | .333 | .25 | .952 | .917 | 2623 | 87 | 18 |

Machine evidence:
[`autotrain-cycle-1859-constraint-graph-null.json`](autotrain-cycle-1859-constraint-graph-null.json).
