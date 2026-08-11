# KERN-06 — Explicit event traces and named machine cost models (SLM-524)

## Claim

Abstract execution cost is a Nat fold of an explicit `Event` trace under a
parameterized `MachineModel.cost`. Named models cover metrics the repo already
records. **No theorem equates abstract cost with physical wall-clock** unless an
explicit external `ThroughputAssumption` / `PhysicalLatencyHypothesis` is
supplied.

## Events (owner-justified only)

| Event | Primary owner |
| --- | --- |
| `tokenize` | `DecodeStats.tokens_emitted` |
| `grammarTransition` | `DecodeStats.dfa_sync_count` |
| `solverExpansion` | `DecodeStats.solver_expanded_nodes` |
| `certificateCheck` | `DecodeStats.solver_verifier_calls` |
| `neuralForward` | `DecodeStats.forwards_count` |
| `neuralBackward` | train-loop `timed("backward")` (unobserved under decode projection) |
| `optimizerStep` | train-loop `timed("optim_step")` (unobserved under decode) |
| `memoryRead` | lex-byte counters exist; omitted from unit-work model (optional) |
| `memoryWrite` | no decode owner → unobserved |
| `kernelLaunch` | no DecodeStats CUDA/host launch counter → unobserved |

## Named models

| Id | Version | Cost rule |
| --- | ---: | --- |
| `DecodeUnitWorkModel` | 1 | payload sum over tokenize/grammar/solver/cert/forward |
| `SolverQueryModel` | 1 | solver expansion + certificate check |
| `NeuralForwardOnlyModel` | 1 | neural forward only (I2 surface) |
| `TrainStepModel` | 1 | forward + backward + optimizer |

## Theorems / API

Lean (`LeverProofLean.EventTrace`): `traceCost`, `traceCost_append`,
`cost_nonneg`, `decodeUnitWork_traceCost_eq_sum`, `wallClockUpperBound`,
`PhysicalLatencyHypothesis`, `physical_bound_from_hypothesis`.

Python (`slm_training.formal.event_trace`): mirrors above;
`project_decode_stats` / `project_decode_stats_record` lossless for represented
counters; `export_four_axis_event_trace_evidence` persists
`resource_cost_model_id = DecodeUnitWorkModel.v1`.

## Empirical boundary

`DecodeStats.*_ms` fields and unobserved event kinds remain empirical remainder.
Wall-clock transfer is only the shape `cost × latency_per_cost_unit` under a
caller-supplied assumption — never a free latency theorem.

## Downstream

INTEG-01 ([integ-01-canonical-proof-trace.md](integ-01-canonical-proof-trace.md)) projects runtime evidence into this Event/cost vocabulary; it does not fork the cost models and does not claim refinement (KERN-11 / [kern-11-proof-trace-refinement.md](kern-11-proof-trace-refinement.md)).
