# Autotrain c1746: classification repair replay remains incomplete

**Verdict:** the exact frozen three-record replay after the v15/v288
classification repair still times out on `smoke_hero_01` in both arms. The
same immutable code completed that record in the preceding one-record probe,
so the remaining gap is a near-wall robustness problem, not permission to
change the timeout. Partial-suite quality is identical and canvas is slower;
canvas is not promoted and the comparison remains unscoreable.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 7,766.83 ms | 16,948.37 ms | 138.360 ms | 141 | 314,046 | 23,926 | 353,964 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 7,920.08 ms | 16,707.92 ms | 141.991 ms | 144 | 314,135 | 24,034 | 354,121 | incomplete |

Rates and completed-document p50 exclude the timed-out record. The inclusive
p50 includes the timeout observation; with three records its median lands on
the slower completed record. AgentV reports one runtime timeout per arm, so
neither row supports a model or ship claim.

## Signals and next run

- The v59 classifier correctly emits `positive=false` and
  `measurement_complete=false`; both arms have count-consistent typed timeout
  evidence and zero unconstrained fallback.
- Candidate completed-document MPR/ms regresses by about 1.94%; it fails the
  5% minimum-effect rule independently of the incomplete measurement.
- Relative to c1745, completed-document p50 improves by about 1.7% for control
  and 2.6% for canvas, and each arm performs more certified work. This is
  harness-throughput progress, not model-quality evidence.
- The clean one-record probe completed `smoke_hero_01` in 23,821.64 ms, but
  this replay reached the 24-second wall. Profile parser accept-set refresh
  and parser-state-key construction next; retain only an exact repair that
  makes completion robust without changing search order, proof budgets,
  grammar authority, or the wall.
- Formal evidence is not applicable to this incomplete screen. The retained
  repair passed 146 exact completion/artifact tests, 226 compiler-decode tests,
  and a fresh Lean build plus axiom audit.

Machine-readable evidence is in
[`autotrain-cycle-1746-canvas-timeout.json`](autotrain-cycle-1746-canvas-timeout.json).
