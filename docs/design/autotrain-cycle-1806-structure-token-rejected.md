# Autotrain c1806: structure-token weighting regresses quality

**Verdict:** reject. Both 20-step CPU scratch arms completed at 1,608,962
parameters and seed 101806. Direct grammar `STRUCT` reconstruction weighting
lowers the measured structure-token CE, but regresses smoke structural
similarity `.1725→.1375` and raises p50 latency `2425.58→7651.84` ms.

| Arm | Loss | Structure | Binder F1 | Recall | MPR | Fidelity | Reward | p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight 0 control | 21.99834 | .1725 | .63333 | 0 | 0 | .52778 | .80733 | 2,425.58 |
| STRUCT weight 1 | 24.82895 | .1375 | .82222 | 0 | 0 | .72222 | .88267 | 7,651.84 |

The objective is active: on the final matched batch, `STRUCT` CE falls
`6.3932→5.6398`. Component CE simultaneously worsens `23.6372→26.2286`, with
61 structure tokens versus 13 component tokens. The binder/fidelity gains do
not override the primary-quality and latency regressions, and neither arm
produces a meaningful program.

The local explicit-no-sync checkpoint SHA-256 values are `d0b5c4cf...f22ae`
(control) and `689ce69c...14c80` (candidate). Neither is reusable, promotable,
synced, or ship evidence. AgentV bundles are complete, fixture gates fail, and
Lean is `not_applicable:screening`.

The next distinct, size-matched hypothesis uses a count-normalized auxiliary
that gives component and `STRUCT` family means equal contribution. It tests
the observed cross-family tradeoff without retuning either rejected
single-family dose, adding parameters, or changing constrained decoding.

Machine evidence:
[`autotrain-cycle-1806-structure-token-rejected.json`](autotrain-cycle-1806-structure-token-rejected.json).
