# E306 component coverage diagnostic (2026-07-17)

E306 extends the canonical `diagnose_eval` report with per-suite component-type
support: gold occurrence/type coverage, unseen types, and types below a
configurable training-frequency floor. The run compares the 480-record E218
train corpus with all five remediated eval suites on CPU; no training occurs.

All suites have 1.0 component occurrence and type coverage, no unseen types,
and no eval type with fewer than five training occurrences. Component scarcity
therefore does not explain E305's held/OOD recall collapse.

The existing lexical report identifies the sharper gap:

| Suite | OpenUI token coverage | Placeholder token coverage |
| --- | ---: | ---: |
| smoke | 0.9500 | 0.8438 |
| held_out | 0.8752 | 0.7606 |
| adversarial | 0.8725 | 0.7805 |
| ood | 0.8678 | 0.7703 |
| rico_held | 0.8390 | 0.8472 |

Metric ceilings remain 1.0 and every p95 fits the 256-token canvas. AgentV
passes 1/1 with zero execution errors.

**Verdict:** do not synthesize components merely because held/OOD recall is
zero; their types already have support. The next data lever should improve
prompt/placeholder lexical diversity and prompt-to-component alignment through
the canonical synthesis pipeline, with leakage checks and judge gating.

Artifacts:

- `outputs/runs/e306-component-coverage-diagnostic/report.json`
- [machine-readable result](component-coverage-results-iter-e306-20260717.json)
