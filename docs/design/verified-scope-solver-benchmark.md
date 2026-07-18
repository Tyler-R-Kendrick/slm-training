# Verified-scope-solver matched matrix & benchmark (VSS4-02, SLM-75)

**Date:** 2026-07-18 · **Status:** fixture wiring landed; frontier rows specified, not run until VSS4-03.
**Code:** `src/slm_training/harnesses/model_build/verified_solver_matrix.py`,
`scripts/run_quality_matrix.py` (`--matrix-set verified-solver`),
`src/slm_training/autoresearch/verified_scope_matrix.py`.
**Consumes:** the VSS4-01 exhaustive finite benchmark
([vss4-01-solver-benchmark-20260718.md](vss4-01-solver-benchmark-20260718.md)).
**Contract:** [verified-scope-solver.md](verified-scope-solver.md).

This is the matched evaluation matrix that lets one existing experiment system
compare every architectural layer of verified scope solving under matched
controls, with correctness/proof gates evaluated before any quality or
search-work gain. It adds schemas, metrics, row definitions, fixture wiring, and
report rendering; it does **not** make a frontier quality claim or require a GPU
run.

## Primary invariant

> The matrix measures correctness authority separately from search efficiency
> and output quality. A row can be faster or more accurate on semantic metrics
> and still fail if it produces one false certified prune, removes one unknown
> candidate, or returns an unverified solved output.

## What's here

* **Metric schema** (stable, zero-default, grouped). Correctness/proof integrity
  (`solver.*` + `surface.semantic_ir_mutation_violations`), exact-search-work,
  capsule/decomposition, topology-diffusion, energy-ranking, surface-realization,
  and the preserved `quality.*` semantic metrics (`parse`, `meaningful`,
  `component`, `dataflow`, `behavior`, exact/near match). The false-support
  fields are available only for closed benchmark cases with independent ground
  truth; elsewhere they are `null` / `not_applicable`, never averaged into zero.
* **Matched rows R0-R6** with declared controls and a single-variable delta each.
* **Eight fail-closed hard gates**, evaluated before quality gains.
* **CPU fixture executor** wrapping the VSS4-01 benchmark; **frontier mode**
  resolves real artifacts and fails clearly when they are unavailable.
* **Typed autoresearch campaign** (`verified-scope-solver`) with four candidate
  hypotheses plus a matched control.

The metric field names mirror their producers so a frontier run populates them
without renaming: `solver_bench.SuiteReport`, `decode_stats.DecodeStats` +
`dsl/solver/closure.py::ClosureCounters`, `dsl/solver/capsule_solver.py::CapsuleCounters`,
`dsl/solver/topology_solver.py`, `models/solver_energy.py::CandidateEnergyRanker`,
`dsl/surface.py::SurfaceRealizationResult`.

## Hard gates (any non-zero ⇒ row fails, before quality is considered)

```text
false_unsupported_count == 0
unknown_preservation_violations == 0
certificate_replay_failures == 0
solved_without_final_verifier == 0
certified_unsat_with_incomplete_proof == 0
candidate_set_parity_failures == 0
surface.semantic_ir_mutation_violations == 0
structured_or_observable_slots_routed_to_ar == 0
```

Fail-closed: on a closed benchmark row a *missing* correctness measurement fails
(it does not pass as zero). The ground-truth-only gates
(`false_unsupported_count`, `unknown_preservation_violations`) are
`not_applicable` on rows without independent ground truth and are excluded from
the pass computation. Every existing OpenUI ship gate (`DEFAULT_SHIP_GATES`) is
retained unchanged; no verified-solver row may weaken
grammar/schema/dataflow/behavior/adversarial/OOD requirements.

## Matched rows

| Row | Method | Control | Single variable | Required capability (frontier) |
| --- | --- | --- | --- | --- |
| R0 | Current matched control — verified solver off, current deterministic finalization | — | baseline | — (fixture) |
| R1 | Exact deterministic solver — compiler-choice exact closure, deterministic ranker | R0 | exact_closure=on | — (fixture: VSS4-01) |
| R2 | Exact solver + existing model ranking | R1 | ranker=model | twotower_ranker_checkpoint |
| R3 | Capsule-aware topology solver (dependency capsules / SCC joint solving) | R2 | capsule_topology=on | capsule_benchmark_family_c |
| R4 | Capsule solver + cost-to-go energy ranking | R3 | ranker=energy | cost_to_go_energy_checkpoint |
| R5 | Deterministic late realization | R3 | late_realization=deterministic | surface_benchmark_family_e |
| R6 | AR late realization (with deterministic fallback) | R5 | realizer=ar | surface_ar_checkpoint |

If a row's required checkpoint/head/capability is absent, it is marked
`blocked/not_run` with a reason; it is never silently substituted with a weaker
configuration under the same row ID.

### Required ablations / strata

* exact closure on/off under the same ranker (R0 vs R1);
* model vs energy ranker over the same exact live sets (R3 vs R4);
* lexical/AST-local decomposition vs dependency capsules (R2 vs R3);
* low/medium/high coupling at matched AST size (within-row strata);
* small/large interface width at matched capsule node count (within-row strata);
* deterministic vs AR realization on the same semantic IR (R5 vs R6);
* alpha-renamed and unseen-identifier held-outs (within-row strata).

## Fixture wiring result (2026-07-18 UTC, CPU, torch-free)

Full JSON (updated in place): [verified-scope-solver-matrix-results.json](verified-scope-solver-matrix-results.json).
Dated memo with the run metadata and honesty caveats:
[vss4-02-matched-matrix-metrics-20260718.md](vss4-02-matched-matrix-metrics-20260718.md).

R0 (control) and R1 (closed exact-search benchmark) run; R2-R6 are blocked with
explicit required-capability reasons. R1 drives the committed VSS4-01 family-A
fixture (`vss4-01/verified_scope_solver/v1`, 4 closed cases): status_counts
`solved 1 · certified_unsat 2 · unknown 1`, `false_unsupported_count 0`,
`unknown_preservation_violations 0`, `certificate_replay_failures 0`, hard gates
**PASS**. No model, quality, or ship claim.

## Verification

```bash
python -m pytest tests/test_scripts/test_quality_matrix_verified_solver.py \
  tests/test_autoresearch/test_verified_scope_matrix.py -q
python scripts/run_quality_matrix.py --matrix-set verified-solver --describe
python scripts/run_quality_matrix.py --matrix-set verified-solver
python -m scripts.repo_policy
```

## Scope / deferred

* **Fixture** runs only the torch-free closed exact-search row (R1) plus the
  control (R0). Model ranking (R2), capsule solving (R3), energy ranking (R4),
  and surface realization (R5/R6) require trained checkpoints or benchmark
  families (capsule family C, surface family E) that are not committed here.
* **Frontier** execution — populating every metric group from the trained
  producers and publishing replayable evidence — is **VSS4-03** (SLM-76).
