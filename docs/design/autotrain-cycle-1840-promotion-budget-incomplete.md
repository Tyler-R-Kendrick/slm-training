# Autotrain c1840: promotion measurement incomplete

**Verdict:** no model conclusion. Lean proved the locked promotion obligation,
and both size-matched arms completed training, but the fixed three-way wall
allocation left only 46.7 seconds per train-plus-eval arm. Both evaluations
stopped before a scoreable smoke or held-out scoreboard was finalized.

| Arm | Params | Train | Loss | Smoke complete | Held-out complete | Partial tokens | Partial forwards |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| capacity-aware margin control | 1,608,962 | 20/20 | 27.4813 | 0/3 | 0/5 | 63 | 14 |
| confirmed tail candidate | 1,608,962 | 20/20 | 33.2064 | 0/3 | 0/5 | 63 | 14 |

Both arms used CPU scratch TwoTower, batch size 2, one thread, seed 101840,
grammar-constrained decoding, and the same 1,608,962 trainable parameters. The
only treatment difference remained tail loss weight 0 versus 1. Control train
took 13.62 seconds and candidate train took 13.87 seconds. The local checkpoint
SHAs are `1d0aa0ec...75f5` and `6834a759...4cfa`, respectively.

The formal preflight proved
`metrics.structural_similarity_monotone` in 6.09 seconds; artifact SHA
`930cba71...82e9`. This integrates the Lean/prover lane before any promotion
training, but proof of the metric contract is not empirical evidence that the
candidate improves the metric.

No parse, fidelity, structure, reward, meaningful-program, exact-AST, or
canonical-equality value is defined. The smoke files contain 0/3 complete
documents and three runtime timeouts per arm; held-out progress contains 0/5
complete documents. Therefore the missing scoreboards are a harness failure,
not a model rejection or promotion failure, and ship remains blocked.

Campaign v138 repairs the canonical allocator: after formal proof completes,
its unused lane is returned symmetrically to the two decision arms while the
15-second finalization reserve and three-minute hard cap remain unchanged. The
frozen retry may reuse these completed train stages only through their
hash-linked manifests; it must rerun both incomplete evaluations and may not
change the locked hypothesis, arms, seed, suites, endpoint, or gates.

Machine evidence:
[`autotrain-cycle-1840-promotion-budget-incomplete.json`](autotrain-cycle-1840-promotion-budget-incomplete.json).
