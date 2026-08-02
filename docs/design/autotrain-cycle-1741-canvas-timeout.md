# Autotrain c1741: clean current-main replay remains incomplete

**Verdict:** the exact frozen replay on squash-merged current `main` still
times out on `smoke_hero_01` in both arms. The two completed documents retain
identical partial-suite quality. Canvas is neither promoted nor rejected, and
its apparent completed-document efficiency delta is below the preregistered
minimum effect as well as being unscoreable while measurement is incomplete.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 9,324.54 ms | 18,897.50 ms | 140.699 ms | 130 | 313,786 | 23,559 | 353,503 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 9,037.41 ms | 18,965.09 ms | 145.633 ms | 129 | 313,727 | 23,495 | 353,407 | incomplete |

Quality rates and completed-document p50 exclude the timed-out record. The
inclusive p50 includes its observed timeout duration. Neither is a complete
authoritative comparison. AgentV reports one runtime timeout in each arm and
ship gates remain blocked.

## Signals and next run

- The v59 classifier correctly emits `positive=false` and
  `measurement_complete=false` from both count-consistent partial scoreboards.
- Candidate completed-document MPR/ms improves by about 3.18%, below the 5%
  minimum effect; the classifier rejects it independently of incompleteness.
- Current-main integration, external-case policy, exact completion regressions,
  and the full Lean build plus axiom audit are green, but the shared timeout
  proves the accepted runtime repairs are still insufficient.
- Next prioritize order-preserving parser-copy allocation below the current
  token-history copy-on-write boundary. Do not transfer branch outcomes,
  reorder witnesses, expand proof budgets, or relax the 24-second wall.
- Formal evidence is not applicable to this incomplete screening comparison;
  every accepted repair still requires a fresh Lean build and axiom audit.

Machine-readable evidence is in
[`autotrain-cycle-1741-canvas-timeout.json`](autotrain-cycle-1741-canvas-timeout.json).
