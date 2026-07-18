# VSS4-02 — Verified-scope-solver metrics for matched matrices (SLM-75)

**Date:** 2026-07-18. **Status:** metric schema + matched rows R0–R6 + fail-closed
correctness gates + CPU fixture runner (consuming the committed VSS4-01 benchmark) +
`--describe`, implemented and tested. **Torch-free fixture path; no learned model,
no network, no long/GPU run, no ship/default-on decision, no frontier quality claim.**
Frontier (model-backed) rows are fully specified but **`not_run` — deferred to
VSS4-03**.

Unblocked by the merged VSS4-01 (SLM-74) benchmark and VSS3 owners
(SLM-70/71/72/73), all on `main`.

## What this issue delivers

Per the ticket, VSS4-02 implements *schemas, metrics, row definitions, fixture
wiring, and report rendering* so that VSS4-03 "can execute the campaign without
inventing metrics, row semantics, or evidence format." It does **not** make a
frontier quality claim or require a long GPU run.

- `src/slm_training/harnesses/experiments/verified_solver_matrix.py`
  - Grouped, **zero-default, backward-compatible** metric schema:
    `SolverProofMetrics` (correctness & proof integrity), `ExactSearchMetrics`,
    `CapsuleMetrics`, `TopologyMetrics`, `EnergyMetrics`, `SurfaceMetrics`, and the
    preserved `QualityMetrics` (existing semantic metrics are **kept, not replaced**).
  - The matched row set **R0–R6** with single-variable deltas and resolved configs
    (`VerifiedSolverRow` records `control_row_id` + the one `variable` it changes).
  - The **fail-closed hard gates** (`evaluate_hard_gates`), evaluated *before* any
    quality/perf comparison.
  - CPU **fixture evaluators** for R0/R1 that consume the VSS4-01 benchmark, and a
    deterministic, config-based `run_id`; shared Markdown rendering.
- `scripts/run_verified_solver_matrix.py` — the `verified-solver` runner
  (`--describe`, `--fixture`, `--out-dir`, JSON+MD evidence, exit-code gate).
- `scripts/run_quality_matrix.py --matrix-set verified-solver [--describe]` — a thin
  guard clause that **delegates to the same functions** and short-circuits the
  quality-training run (no parallel report format; reuses the existing conventions).

## Correctness authority is separate from quality

A row may be faster or score better on semantic metrics and still **fail** the
matrix. The fail-closed gates (each must be exactly zero; `null`/`not_applicable`
observations are never coerced to zero, never averaged, and never a silent pass):

```
false_unsupported_count == 0
unknown_preservation_violations == 0
certificate_replay_failures == 0
solved_without_final_verifier == 0
certified_unsat_with_incomplete_proof == 0
candidate_set_parity_failures == 0
semantic_ir_mutation_violations == 0
structured_or_observable_slots_routed_to_ar == 0
```

Every existing ship gate is retained; no new row weakens grammar/schema/dataflow/
behavior/adversarial/OOD requirements. A fixture-only row proves wiring but cannot
satisfy a frontier ship gate.

## Matched rows (single-variable deltas)

| row | variable | control | fixture |
| --- | -------- | ------- | ------- |
| R0 | baseline (verified solver off) | — | **run** |
| R1 | exact_closure_on (deterministic ranker/realizer) | R0 | **run** |
| R2 | ranker=model | R1 | not_run (frontier) |
| R3 | decomposition=capsule_topology | R1 | not_run (frontier) |
| R4 | ranker=energy (candidate-set parity asserted) | R3 | not_run (energy head) |
| R5 | late_realization=deterministic | R1 | not_run (frontier) |
| R6 | late_realization=ar | R5 | not_run (surface head) |

A model-backed row whose checkpoint/head is absent is marked **`not_run` with a
reason** — never silently substituted with a weaker config under the same row id.

## Fixture evidence (CPU, VSS4-01 benchmark, independent ground truth)

R1 runs the exact solver over the closed VSS4-01 word-tree fixture (candidate `a`
supported, `b`/`c` certified-unsat, `d` unknown):

- `status_counts` = solved 1 / certified_unsat 2 / unknown 1
- `false_unsupported_count` = **0** (measured against ground truth, not `null`)
- `unknown_preservation_violations` = 0 · `certificate_replay_failures` = 0
- 4/4 certificates emitted and replayed; real `SearchCounters` recorded.

R0 (solver off) reports every solver correctness field as `null`
(`not_applicable`) and makes no correctness claim. Committed report:
`docs/design/vss4-02-verified-solver-matrix-results.json`
(run_id `aaad08463362586a`, all gates pass).

## Verification

```bash
python -m pytest tests/test_harnesses/experiments/test_verified_solver_matrix.py -q  # 19 passed
python -m pytest tests/test_harnesses/experiments/ tests/test_harnesses/test_solver_bench.py -q  # 52 passed
python -m pytest tests/ -q -k quality_matrix                                          # 11 passed
python -m scripts.run_verified_solver_matrix --describe                               # passed, 0 gate failures
python -m scripts.run_verified_solver_matrix --fixture                                # passed, 0 gate failures
python scripts/run_quality_matrix.py --matrix-set verified-solver --describe          # passed (no model/data loaded)
python -m ruff check src/slm_training/harnesses/experiments/verified_solver_matrix.py scripts/run_verified_solver_matrix.py
python -m scripts.repo_policy
```

Tests cover: schema zero/default/JSON-scalar/backward-compatibility; R1 closed-benchmark
false-unsupported/unknown-preservation exactness; R0 `not_applicable` (not zero); all
seven matched rows resolve to single-variable deltas; deterministic config `run_id`;
model-backed rows `not_run` (not silently downgraded); each hard gate fails on its
injected violation (false prune, unknown removal, replay failure, unverified solved,
candidate-set mismatch, semantic-IR mutation, structured-string AR routing);
`not_applicable` distinct from a measured pass; and fixture JSON/Markdown consistency.

## Scope / deferred to VSS4-03

The capsule/topology/energy/surface metric **groups** are stable zero-default schema
fields here; populating them from live capsule/topology/energy/surface runs, executing
the model-backed rows R2–R6 under matched controls, the autoresearch campaign
execution, and any frontier quality/perf claim are **VSS4-03**. Every frontier row is
labeled `not_run` and no claim is derived from `--describe` or the fixture wiring.
