# Autotrain c1744: v13 replay remains incomplete

**Verdict:** the exact frozen replay after caching semantic tokenizer
projections still times out on `smoke_hero_01` in both arms. The repair lowers
completed-document latency and advances more certified work, but the two
completed documents retain identical partial-suite quality. Canvas is neither
promoted nor rejected; its within-cycle efficiency delta is below the
preregistered minimum and the comparison remains unscoreable.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,263.62 ms | 18,155.82 ms | 175.557 ms | 129 | 313,735 | 23,501 | 353,421 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,231.85 ms | 18,275.56 ms | 144.462 ms | 134 | 313,834 | 23,686 | 353,592 | incomplete |

Rates and completed-document p50 exclude the timed-out record. The inclusive
p50 includes its observed timeout duration. AgentV reports one runtime timeout
per arm, so neither row supports a model or ship claim.

## Signals and next run

- The v59 classifier correctly emits `positive=false` and
  `measurement_complete=false` for both count-consistent partial scoreboards.
- Candidate completed-document MPR/ms improves by about 0.39%, below the 5%
  minimum effect and rejected independently of incompleteness.
- Relative to c1743, completed-document p50 improves by about 5.0% in both
  arms, while the frozen one-record profile advances from 76 to 79 tokens.
  This is exact harness-throughput progress, not model-quality evidence.
- Next profile immutable `SemanticState` and frame construction. The prior
  cProfile attributes 1.330 seconds to 68,919 `dataclasses.replace` calls;
  test direct constructors only if exact parity and timeout propagation remain
  green. Do not change search order, proof budgets, grammar authority, or the
  24-second wall.
- Formal evidence is not applicable to this incomplete screen. The retained
  v13 repair passed 145 exact completion tests and a fresh Lean build plus
  axiom audit.

Machine-readable evidence is in
[`autotrain-cycle-1744-canvas-timeout.json`](autotrain-cycle-1744-canvas-timeout.json).
