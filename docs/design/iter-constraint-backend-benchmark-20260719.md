# SDE3-04 Constraint-Backend Benchmark Manifest

- **manifest_id**: `iter-constraint-backend-benchmark-20260719`
- **schema_version**: `constraint_backend_benchmark/v1`
- **hypothesis_id**: `H17`
- **activation_status**: `activation_blocked`
- **activation_verdict**: `activation_blocked`
- **campaign_verdict**: `unrun`
- **primary_metric**: `binding_aware_meaningful_program_rate`
- **seeds**: [0, 1, 2]
- **null_threshold_percent**: 5.0
- **manifest_hash**: `b2e3c515418dd20a`

## Activation gates

| gate_id | depends_on_issue_id | required_status | available | evidence |
|---|---|---|---|---|
| eval_cache_or_cost_approved | SLM-175 | Done | False |  |
| budget_approved |  |  | False |  |

## Backends

| backend_id | package_name | package_version | local_offline | supported_kinds |
|---|---|---|---|---|
| current | openui_current | repo | True | openui |
| syncode | syncode | unset | True | - |
| domino | domino | unset | True | - |
| xgrammar | xgrammar | unset | True | - |
| unconstrained | unconstrained | repo | True | - |

## Budget caps

- microbenchmark_repetitions: 0
- end_to_end_repetitions: 0
- max_dollars: 0.0
- gpu_hours: 0.0

## Arms

| arm_id | backend_id | benchmark_layer | eligible | omission_reason |
|---|---|---|---|---|
| current_static_micro | current | static_micro | True |  |
| current_language_equivalence | current | language_equivalence | True |  |
| current_end_to_end_surface | current | end_to_end_surface | True |  |
| syncode_static_micro | syncode | static_micro | True |  |
| syncode_language_equivalence | syncode | language_equivalence | True |  |
| syncode_end_to_end_surface | syncode | end_to_end_surface | True |  |
| domino_static_micro | domino | static_micro | True |  |
| domino_language_equivalence | domino | language_equivalence | True |  |
| domino_end_to_end_surface | domino | end_to_end_surface | True |  |
| xgrammar_static_micro | xgrammar | static_micro | True |  |
| xgrammar_language_equivalence | xgrammar | language_equivalence | True |  |
| xgrammar_end_to_end_surface | xgrammar | end_to_end_surface | True |  |
| unconstrained_static_micro | unconstrained | static_micro | True |  |
| unconstrained_language_equivalence | unconstrained | language_equivalence | True |  |
| unconstrained_end_to_end_surface | unconstrained | end_to_end_surface | True |  |

## Note

SDE3-04 constraint-backend benchmark manifest (wiring slice). No decoder package is installed and no benchmark is run.

## Version stamp

```json
{
  "stamp_schema": "version_stamp/v1",
  "code_commit": "d11405a48c39d8b52dae26be6cfd763d3d0ca5c4",
  "code_dirty": true,
  "components": {
    "harness.experiments": "v11",
    "matrix.perf": "v1"
  },
  "stamped_at": "2026-07-19T18:21:40.221311+00:00"
}
```
