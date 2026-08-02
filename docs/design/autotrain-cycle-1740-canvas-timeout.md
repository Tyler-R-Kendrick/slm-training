# Autotrain c1740: v8 replay remains incomplete

**Verdict:** the exact frozen replay after the v8 parser-allocation repair still
times out on `smoke_hero_01` in both arms. The two completed documents retain
the same partial-suite quality. Canvas is neither promoted nor rejected, and
the apparent completed-document efficiency delta is below the preregistered
minimum effect as well as being unscoreable while measurement is incomplete.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 9,395.42 ms | 19,078.93 ms | 137.845 ms | 134 | 313,834 | 23,686 | 353,592 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 9,030.90 ms | 18,827.20 ms | 141.260 ms | 131 | 313,793 | 23,587 | 353,518 | incomplete |

Quality rates and completed-document p50 exclude the timed-out record. The
inclusive p50 includes its observed timeout duration. Neither is a complete
authoritative comparison. AgentV reports one runtime timeout in each arm and
ship gates remain blocked.

## Signals and next run

- The harness correctly emits `positive=false` and
  `measurement_complete=false` for both partial scoreboards.
- Candidate completed-document MPR/ms improves by about 4.04%, below the 5%
  minimum effect; the classifier rejects it independently of incompleteness.
- The v8 allocation repair is exact-parity and Lean validated, but the shared
  timeout proves it is not sufficient by itself.
- Next prioritize mechanical list/cache allocations inside control-only parser
  copies. Do not transfer branch outcomes, change traversal order, expand
  budgets, or relax the 24-second wall.
- Formal evidence is not applicable to this incomplete screening comparison;
  every accepted repair still requires a fresh Lean build and axiom audit.

Machine-readable evidence is in
[`autotrain-cycle-1740-canvas-timeout.json`](autotrain-cycle-1740-canvas-timeout.json).
