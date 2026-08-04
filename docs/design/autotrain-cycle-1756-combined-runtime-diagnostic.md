# Autotrain c1756: combined bounds and canvas diagnostic

**Verdict:** reject. Combining completion bounds with compact active canvas is
an exact smoke and held-out quality/loss null. It moves latency in opposite
directions: 1.90% faster on smoke and 7.79% slower on held-out.

| Arm | Params | Suite n | Parse | Binder F1 | Meaning | Structure | Recall | p50 | Loss / train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| control | 1,608,962 | smoke 3 / held 5 | 1 / 1 | .6333 / .4371 | .3333 / .20 | .13527 / .10690 | .1667 / .1286 | 1,284.48 / 1,205.19 ms | 12.5226 / 3.900 s |
| bounds + canvas | 1,608,962 | smoke 3 / held 5 | 1 / 1 | .6333 / .4371 | .3333 / .20 | .13527 / .10690 | .1667 / .1286 | 1,260.02 / 1,299.04 ms | 12.5226 / 2.796 s |

The size-matched 21-step CPU scratch arms used seed 101756, batch size 2,
strict compiler-tree constrained decoding, and candidate-first/control-second
execution. Every smoke and held-out document completed with zero decode
timeouts. The candidate saves 28.29% train wall, but the suite-dependent decode
latency and exact quality null do not satisfy a quality-aware promotion.

AgentV completed without execution errors. Both `--ship-gates` verdicts fail on
fixture volume and quality; adversarial, OOD, and `rico_held` were not run. No
champion exists, so Lean is `not_applicable:no_champion`, RL remains locked, and
neither local checkpoint may be reused, promoted, synced, or shipped.

Next: the preregistered steps diagnostic under the same size-matched controls,
testing whether added training cost changes held-out structure rather than
assuming a loss or latency benefit.

Machine evidence:
[`autotrain-cycle-1756-combined-runtime-diagnostic.json`](autotrain-cycle-1756-combined-runtime-diagnostic.json).
