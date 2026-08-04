# Autotrain c1747: frozen replay completes and rejects canvas

**Verdict:** the exact frozen three-record control/canvas comparison completed
without a timeout after the v16 completion-kernel repair. Quality and exact
work counters are identical. Compact canvas is 63.95 ms slower at p50, so the
candidate is rejected; the run is fixture screening evidence, not a ship or
model-quality claim.

## Result matrix

| Arm | Params | Complete | Timeout | Parse | Meaning | Structure | Binder F1 | p50 | Init | Forwards | Tokens | Compiler ms | States | Witnesses | Forks | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| matched control | 1,608,962 | 3/3 | 0 | 1.000 | 0.667 | 0.16207 | 1.000 | 16,518.49 ms | 147.716 ms | 149 | 244 | 34,853.274 | 314,142 | 24,066 | 354,139 | baseline |
| compact canvas | 1,608,962 | 3/3 | 0 | 1.000 | 0.667 | 0.16207 | 1.000 | 16,582.44 ms | 139.618 ms | 149 | 244 | 34,926.022 | 314,142 | 24,066 | 354,139 | rejected |

AgentV emitted both completed evaluations with zero execution errors. Honest
ship gates still fail: `n=3` is below the smoke minimum, structural similarity
is below 0.35, and AST/canonical BEq are zero. Parse, meaningful-program,
component-recall, binder, placeholder, reward, no-fallback, and no-timeout
checks pass.

## Signals and next run

- The prior infrastructure blocker is resolved: both arms complete 3/3 under
  the unchanged 24-second document wall, versus 2/3 in c1746.
- Identical candidates, forwards, tokens, states, witnesses, and forks show
  compact canvas has no effect on this frozen compiler-tree workload.
- Candidate p50 efficiency regresses by about 0.386%, far below the 5%
  minimum-effect threshold. The lower aggregate total is timing noise and does
  not override the preregistered p50 decision.
- The next hypothesis is the distinct, size-matched `component-plan` model
  lever, with the matched control retained. RL remains locked because no arm is
  ship-eligible.
- Lean/LeverProof does not calibrate this empirical fixture metric, so no band
  disposition is inferred. The integrated v16 repair passed the 26-job
  LeverProof build/test and the 2,947-job formal contract/axiom audit before
  this replay; formal validation remains a required promotion gate.

Machine-readable evidence is in
[`autotrain-cycle-1747-frozen-replay.json`](autotrain-cycle-1747-frozen-replay.json).
