# Autotrain c1742: v11 replay remains incomplete

**Verdict:** the exact frozen replay after copy-on-write tokenizer-map sharing
still times out on `smoke_hero_01` in both arms. The two completed documents
retain identical partial-suite quality. Canvas is neither promoted nor
rejected; its apparent efficiency delta is below the preregistered minimum and
the comparison remains unscoreable.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,848.61 ms | 18,834.58 ms | 138.062 ms | 130 | 313,735 | 23,501 | 353,421 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,637.31 ms | 19,042.76 ms | 145.334 ms | 129 | 313,735 | 23,501 | 353,421 | incomplete |

Rates and completed-document p50 exclude the timed-out record. The inclusive
p50 includes its observed timeout duration. AgentV reports one runtime timeout
per arm, so neither row supports a model or ship claim.

## Signals and next run

- The v59 classifier correctly emits `positive=false` and
  `measurement_complete=false` for both count-consistent partial scoreboards.
- Candidate completed-document MPR/ms improves by about 2.45%, below the 5%
  minimum effect and rejected independently of incompleteness.
- The v11 map-cache repair is exact-parity and Lean validated but insufficient.
- Next test sharing the dormant lexer thread across callback-free control
  descendants while keeping parser stacks independently copied. Do not change
  candidate order, proof budgets, grammar authority, or the 24-second wall.
- Formal evidence is not applicable to this incomplete screen; accepted
  repairs still require a fresh Lean build and axiom audit.

Machine-readable evidence is in
[`autotrain-cycle-1742-canvas-timeout.json`](autotrain-cycle-1742-canvas-timeout.json).
