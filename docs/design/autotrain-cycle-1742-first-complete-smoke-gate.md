# Autotrain c1742: first complete smoke scoreboard, expected fixture reject

**Verdict:** exact frozen replay of the c1741 control/bounds arms after the
AgentV SDK repair. Both arms now produce a complete `scoreboard.json` with
`gates`, and both fail honest ship gates on fixture volume and quality
thresholds as expected for a 3-record smoke suite. This is a routine fixture
diagnostic, not a model regression, and does not stop the loop.

## Result matrix

| Arm | Params | n | parse | binder F1 | meaningful | structure | p50 (ms) | Gates | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| control | 1,608,962 | 3 | 1.0 | 0.6333 | 0.0 | 0.0575 | 1,610.57 | fail | rejected (fixture) |
| bounds (candidate) | 1,608,962 | 3 | 1.0 | 0.6333 | 0.0 | 0.0575 | 1,653.04 | fail | rejected (fixture) |

Gate failures: `insufficient_n` (3 < 20), `meaningful_program_rate` (0.0 <
0.66), `structural_similarity` (0.0575 < 0.35), `component_type_recall`,
`ast_beq_rate`, `canonical_beq_rate`, `reward_score`, plus `missing_suite` for
`held_out`/`adversarial`/`ood`/`rico_held`. Primary endpoint
`smoke.structural_similarity` improvement is 0.0 — control and candidate are
identical (matched-lever screening arms), so this is `fixture_insufficient_n`
non-positive, not a null model delta.

## Signals and next run

- Infrastructure attribution from c1741 is resolved: the AgentV SDK repair
  produced a complete, scoreable comparison.
- Ranked successor: test the size-matched `component-plan` quality hypothesis
  next (`c20260802-continuous-openui-local-8c0b60dd-c2-component-plan`),
  keeping the matched control as baseline every cycle.
- Fixture `n`/quality-threshold fails on a 3-record smoke suite are expected
  diagnostics per the continuous-loop contract; they never terminate the loop.

No checkpoint was promoted; `docs/MODEL_CARD.md` / README are unchanged.
Machine-readable evidence is in
[`autotrain-cycle-1742-first-complete-smoke-gate.json`](autotrain-cycle-1742-first-complete-smoke-gate.json).
