# Autotrain c1743: component-plan candidate regresses vs. control

**Verdict:** the `component-plan` quality lever underperforms its size-matched
control on this 3-record smoke fixture — both binder F1 and the primary
structural-similarity endpoint regress. Combined with `insufficient_n`, this
is a non-positive fixture screening result; `component-plan` is rejected for
this arm, not promoted.

## Result matrix

| Arm | Params | n | parse | binder F1 | meaningful | structure | p50 (ms) | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| control | 1,755,760 | 3 | 1.0 | 0.7333 | 0.3333 | 0.2308 | 3,808.12 | rejected (fixture) |
| component-plan | 1,755,760 | 3 | 1.0 | 0.6333 | 0.0 | 0.1725 | 3,734.80 | rejected (fixture + regression) |

`smoke.structural_similarity` improvement is **-0.0583** (control 0.2308 →
candidate 0.1725); `binder_reference_f1` regresses 0.7333 → 0.6333. Both fail
`insufficient_n` (3 < 20) independently of the regression.

## Signals and next run

- `non_regression_fail:binder_reference_f1` fired — the harness correctly
  blocks a lever that measurably regresses a tracked quality metric, even on
  a fixture-sized suite.
- Ranked successor: test the distinct size-matched `component-edge` quality
  hypothesis next (`c20260802-continuous-openui-local-8c0b60dd-c3-component-edge`),
  keeping the matched control as baseline every cycle.
- No harness defect implicated; this is model-lever screening evidence, not
  an infrastructure signal.

No checkpoint was promoted; `docs/MODEL_CARD.md` / README are unchanged.
Machine-readable evidence is in
[`autotrain-cycle-1743-component-plan-regression.json`](autotrain-cycle-1743-component-plan-regression.json).
