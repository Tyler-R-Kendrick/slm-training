# Autotrain c1745: v14 replay remains incomplete

**Verdict:** the exact frozen replay after replacing hot immutable semantic
state/frame copies with direct constructors still times out on `smoke_hero_01`
in both arms. The repair lowers completed-document latency relative to c1744,
but the two completed documents retain identical partial-suite quality and the
canvas arm is slower within this cycle. Canvas is not promoted; the comparison
remains unscoreable.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 complete | p50 incl. incomplete | Init | Forwards | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 7,897.19 ms | 17,712.37 ms | 150.168 ms | 137 | 313,900 | 23,770 | 353,708 | incomplete |
| compact canvas | 1,608,962 | 2/3 | 1 | 1.000 | 0.500 | 0.11665 | 1.000 | 8,130.19 ms | 17,855.21 ms | 176.644 ms | 138 | 313,916 | 23,794 | 353,736 | incomplete |

Rates and completed-document p50 exclude the timed-out record. The inclusive
p50 includes its observed timeout duration. AgentV reports one runtime timeout
per arm, so neither row supports a model or ship claim.

## Signals and next run

- The v59 classifier correctly emits `positive=false` and
  `measurement_complete=false` for both count-consistent partial scoreboards.
- Candidate completed-document MPR/ms regresses by about 2.87%; it fails the
  5% minimum-effect rule independently of the incomplete measurement.
- Relative to c1744, completed-document p50 improves by about 4.4% for control
  and 1.2% for canvas, while more certified tokens are emitted. This is exact
  harness-throughput progress, not model-quality evidence.
- Re-profile the current v14/v287 runtime before changing code. Prior evidence
  points toward semantic-state interning or compiler decision classification,
  but either hypothesis must first clear the measured hot-path threshold and
  demonstrate reuse. Do not change search order, proof budgets, grammar
  authority, or the 24-second wall.
- Formal evidence is not applicable to this incomplete screen. The retained
  v14 repair passed 145 exact completion tests and a fresh Lean build plus
  axiom audit.

Machine-readable evidence is in
[`autotrain-cycle-1745-canvas-timeout.json`](autotrain-cycle-1745-canvas-timeout.json).
