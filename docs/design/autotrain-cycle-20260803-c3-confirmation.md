# Autotrain cycle c3 confirmation

Campaign: `continuous-loop-20260803-continuous-openui-202607-98199209-c3`  
Loop: `continuous-openui-20260730`  
Recipe: CPU, scratch backend, 20 steps, 1,755,764 trainable parameters, strict grammar-constrained compiler-tree mode, smoke `n=3`, local-only scratch checkpoint.

| Arm | Structural similarity | Meaningful rate | Component recall | Binder F1 | Parse | p50 latency | Ship |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Control | 0.2308 | 0.3333 | 0.1667 | 0.4889 | 1.0000 | 6944.49 ms | Fail |
| Candidate | 0.2308 | 0.3333 | 0.1667 | 0.2667 | 1.0000 | 7201.72 ms | Fail |

The fresh-seed confirmation rejected the c2 champion: primary quality did not re-hold and binder F1 regressed by 0.2222. Both arms are fixture-only and below the smoke evidence floor (`n=3 < 20`); held-out, adversarial, OOD, and RICO suites were not run. No promotion, RL unlock, checkpoint sync, or stack layer was opened.

Next-run steering is to exhaust this fingerprint, preregister a distinct size-matched structural/meaningful-quality objective, keep loss diagnostic rather than promotional, and use the next batch-size arm only for runtime diagnosis. Lean remains a fail-closed diagnostic gate; the host has no live Lake binary, so no formal ship claim is made.

Canonical JSON: [`autotrain-cycle-20260803-c3-confirmation.json`](autotrain-cycle-20260803-c3-confirmation.json).
