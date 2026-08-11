# INTEG-04 — Opt-in four-axis ledger adapters (SLM-529)

## Claim

Existing harness families may optionally project EVID-03 four-axis formal
analysis (plus KERN-12 computability labels and INTEG-01 / KERN-11 refinement
refs) into their **canonical report/evidence seams**. Defaults stay unchanged.
Absent formal evidence is recorded as `absent` / `unknown` — never as failure
or proof. Enabling the projection does not change model or solver behavior.

## Adapter (project into, do not fork)

| Surface | Role |
| --- | --- |
| `harnesses/four_axis_ledger.py` | Shared opt-in projection + authority-ceiling guard |
| `harnesses/reasoning/revmath/report.py` / `RevmathReportV1` | Reasoning family seam (`enable_four_axis_ledger=False` default) |
| `harnesses/train_data/report.py` | Non-reasoning `quality_report.json` seam |
| `attach_to_evaluation_report_metadata` | Non-reasoning `model_build` `EvaluationReport.metadata` seam |

Schema: `harness_four_axis_ledger_projection/v1`. Reuses
`RevmathFourAxisAnalysisV1` / `FormalPreflightFourAxisLedgerV1` vocabulary —
no parallel revmath reporting stack and no campaign-store link schema.

## Axes attached (optional refs only)

| Projection slot | Source |
| --- | --- |
| `logical_strength` | `assumption_strength` axis / RM subsystem id |
| `computability` | computability axis + optional KERN-12 class id |
| `resource_bound` | resource_bounds axis / `bound_ast_id` |
| `refinement_empirical_remainder` | implementation_refinement + optional INTEG-01 trace / KERN-11 cert |

`authority_ceiling` is derived from underlying axis status and any supplied
formal authority class. Harnesses **cannot** claim a stronger class than that
ceiling (fail closed).

## Acceptance

- Default-off reports omit `four_axis_ledger_projection` (backward-readable).
- Enabling projection does not mutate solver judgments or training loops.
- Parity fixtures cover `reasoning/revmath`, `train_data`, and `model_build`.

## Tests / fixtures

- `tests/test_harnesses/test_four_axis_ledger_adapters.py`
- `resources/harnesses/integ04_four_axis_ledger_fixtures.v1.json`
