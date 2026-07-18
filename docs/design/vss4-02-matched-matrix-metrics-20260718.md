# VSS4-02 — Matched verified-scope-solver matrix metrics + hard gates (SLM-75)

Date: 2026-07-18 · Track: Verified Scope Solving & Hybrid Realization · Linear: SLM-75

This memo records the fixture-wiring run for the VSS4-02 matched evaluation
matrix. It is **wiring evidence only**: the closed exact-search row runs the
VSS4-01 benchmark on CPU; every frontier row is fully specified but **not run
until VSS4-03**. No model, quality, or ship claim is made here.

## What shipped

* `src/slm_training/harnesses/model_build/verified_solver_matrix.py`
  — the `verified-solver` matrix set: the stable metric schema
  (`SolverCorrectness`, `ExactSearchWork`, `CapsuleMetrics`, `TopologyMetrics`,
  `EnergyMetrics`, `SurfaceMetrics`, `QualityMetrics`), the matched R0-R6 rows
  (`verified_solver_rows`), deterministic config hashing (`config_hash`), the
  eight fail-closed hard gates (`evaluate_verified_solver_gates`), the CPU
  fixture executor (`run_fixture_row`, wrapping
  `slm_training.harnesses.solver_bench.run_reference_suite`), and JSON/Markdown
  rendering (`run_matrix`, `render_markdown`, `describe_rows`).
* `scripts/run_quality_matrix.py` — `--matrix-set verified-solver` dispatch with
  `--describe`, `--row`, and `--mode {fixture,frontier}`. No fourth runner and
  no parallel report format: the existing quality runner owns the new set.
* `src/slm_training/autoresearch/verified_scope_matrix.py` — the typed
  campaign/hypothesis matrix (`verified-scope-solver` / `-m1`) through the
  existing autoresearch schema, grounded in this memo, the contract doc, and the
  fixture results below.

## Metric schema (grouped, zero-default)

The schema separates correctness authority from search efficiency and output
quality. A row can be faster or more accurate and still fail if it produces one
false certified prune, removes one unknown candidate, or returns an unverified
solved output. Groups: `solver.*` (correctness/proof integrity),
`exact_search_work.*`, `capsule.*`, `topology.*`, `energy.*`, `surface.*`, and
the preserved `quality.*` semantic metrics (`parse`, `meaningful`, etc.). The
false-support fields (`solver.false_unsupported_count/rate`,
`solver.unknown_preservation_violations`) are `null` / `not_applicable` on rows
without independent ground truth and are never averaged into zero.

## Hard gates (fail-closed, evaluated before quality gains)

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

Every existing OpenUI ship gate (`DEFAULT_SHIP_GATES` in
`harnesses/model_build/ship_gates.py`) is retained unchanged; no verified-solver
row may weaken grammar/schema/dataflow/behavior/adversarial/OOD requirements. A
fixture-only row can prove wiring but cannot satisfy a frontier ship gate.

## Matched rows (R0-R6)

| Row | Method | Control | Single variable | Fixture |
| --- | --- | --- | --- | --- |
| R0 | Current matched control (solver off) | — | baseline | ran |
| R1 | Exact deterministic solver | R0 | exact_closure=on | ran (closed benchmark) |
| R2 | Exact solver + model ranking | R1 | ranker=model | blocked (twotower_ranker_checkpoint) |
| R3 | Capsule-aware topology solver | R2 | capsule_topology=on | blocked (capsule_benchmark_family_c) |
| R4 | Capsule solver + cost-to-go energy | R3 | ranker=energy | blocked (cost_to_go_energy_checkpoint) |
| R5 | Deterministic late realization | R3 | late_realization=deterministic | blocked (surface_benchmark_family_e) |
| R6 | AR late realization | R5 | realizer=ar | blocked (surface_ar_checkpoint) |

Blocked rows are fully specified and marked `blocked/not_run` with an explicit
required-capability reason; none is silently substituted with a weaker config.

## Measured fixture result (R1 closed exact-search benchmark)

Full JSON: [verified-scope-solver-matrix-results.json](verified-scope-solver-matrix-results.json).

R1 runs the committed VSS4-01 family-A benchmark
(`vss4-01/verified_scope_solver/v1`, manifest digest `1b0a2754d25d4c8d`, 4 closed
cases with independent brute-force ground truth):

| solver metric | value |
| --- | --- |
| status_counts | solved 1 · certified_unsat 2 · unknown 1 · budget_exhausted 0 |
| false_unsupported_count | 0 |
| unknown_preservation_violations | 0 |
| certificates_emitted / replayed | 4 / 4 |
| certificate_replay_failures | 0 |
| solved_without_final_verifier | 0 |
| certified_unsat_with_incomplete_proof | 0 |
| **hard gates (R0, R1)** | **PASS** |

R0 reports every false-support field as `not_applicable` (no solver, no ground
truth). Config hashes are deterministic: R0 `f01ce7639f1ccbdc`, R1
`f81c4ed47b6aa378`.

### Run metadata

| device | steps | backend | matrix set | n (cases) | honesty | gate |
| --- | --- | --- | --- | --- | --- | --- |
| cpu | — (no training) | scratch/torch-free | verified-solver | 4 | fixture_wiring | PASS |

### Reproduction

```bash
python -m pytest tests/test_scripts/test_quality_matrix_verified_solver.py \
  tests/test_autoresearch/test_verified_scope_matrix.py -q
python scripts/run_quality_matrix.py --matrix-set verified-solver --describe
python scripts/run_quality_matrix.py --matrix-set verified-solver
python -m scripts.repo_policy
```

`--describe` loads no checkpoint/data and writes no file. The fixture run writes
`verified_solver_matrix_results.{json,md}` under the run root and mirrors the
JSON to `docs/design/verified-scope-solver-matrix-results.json`.

## Tradeoffs and caveats

* **No frontier claim.** R2-R6 require trained checkpoints / a benchmark family
  (capsule family C, surface family E) that are not committed; they are blocked,
  not approximated. The VSS4-03 campaign executes them.
* **MODEL_CARD is intentionally NOT updated** — no checkpoint was created or
  promoted.
* The `status_counts` mapping treats a VSS4-01 support verdict as its solve-
  status analog (supported→solved, unsupported→certified_unsat, unknown→unknown)
  under the documented correspondence in `verified_solver_matrix.py`; the full
  `SearchStatus` controller counts populate only at frontier scale.
* Exact-search-work counters beyond what the reference support oracle exercises
  stay at honest zero-defaults in fixture mode.
