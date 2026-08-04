# Autotrain c1743: v12 replay remains incomplete

**Verdict:** the exact frozen replay after sharing the dormant lexer thread
still times out on `smoke_hero_01` in both arms. The two completed documents
retain identical partial-suite quality. Canvas is neither promoted nor
rejected; its apparent efficiency delta is below the preregistered minimum and
the comparison remains unscoreable.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,700.31 ms | 18,438.87 ms | 171.880 ms | 128 | 313,716 | 23,474 | 353,387 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,685.98 ms | 18,356.60 ms | 153.018 ms | 128 | 313,716 | 23,474 | 353,387 | incomplete |

Rates and completed-document p50 exclude the timed-out record. The inclusive
p50 includes its observed timeout duration. AgentV reports one runtime timeout
per arm, so neither row supports a model or ship claim.

## Signals and next run

- The v59 classifier correctly emits `positive=false` and
  `measurement_complete=false` for both count-consistent partial scoreboards.
- Candidate completed-document MPR/ms improves by about 0.16%, below the 5%
  minimum effect and rejected independently of incompleteness.
- The v12 lexer-thread repair is exact-parity and Lean validated but
  insufficient; equal state, witness, fork, forward, and token counts localize
  the remaining capacity problem to shared completion/parser work.
- Next profile mutable control-parser stack/value copying at each fork and test
  a copy-on-write representation only if the profile confirms material cost.
  Do not change candidate order, proof budgets, grammar authority, or the
  24-second wall.
- Formal evidence is not applicable to this incomplete screen; accepted
  repairs still require a fresh Lean build and axiom audit.

Machine-readable evidence is in
[`autotrain-cycle-1743-canvas-timeout.json`](autotrain-cycle-1743-canvas-timeout.json).
