# VSS4-01 — Exhaustive finite solver benchmark (SLM-74)

**Date:** 2026-07-18. **Status:** benchmark harness + independent ground-truth
enumerator + committed v1 finite fixture (family A) + CLI, implemented and tested.
Torch-free; **no learned model, no network, no large committed corpus.** This
validates finite-fixture implementation invariants, **not** frontier
generalization or ship quality.

Unblocked by the just-merged SLM-71 (VSS3-03); its VSS0–VSS3 blockers (SLM-64/66/67/72)
are all on `main`.

## Primary invariant

> Every candidate labeled **unsupported** by the evaluated solver is absent from
> the benchmark's independently enumerated verifier-accepted solution set for the
> same pack, constraint version, and finite bounds.

Two deterministic paths decide each case and must agree on every closed fixture:

1. the reference `EnumerativeSupportOracle` (VSS0-04) — the runtime under test;
2. a **benchmark-only brute-force transition enumerator** (`enumerate_ground_truth`),
   written directly against the `ProblemExpander`/`Verifier` protocols — it walks
   every `(hole, value)` transition, applies the pack verifier at each terminal, and
   records the accepted-terminal digest set plus whether coverage was complete. It
   shares none of the oracle's search/certificate internals.

## What's here

- `src/slm_training/harnesses/solver_bench.py` — schema (`SolverBenchmarkCase`),
  independent enumerator, per-case cross-check (`run_case`), suite + deterministic
  order-independent manifest (`run_suite`), and the committed v1 fixture
  (`build_reference_fixture`).
- `scripts/run_solver_bench.py` — thin CLI (`--describe`, `--all`, JSON out;
  non-zero exit on any hard failure).
- `tests/test_harnesses/test_solver_bench.py` — 6 tests.

## Hard failure conditions (any ⇒ suite fails, CLI exits non-zero)

- **false certified prune** — oracle `unsupported` but ground truth has an accepted
  terminal in the candidate's subtree;
- **unknown-preservation violation** — oracle `unsupported` where ground truth is
  `unknown` (incomplete coverage);
- **certificate replay failure**;
- **disagreement** — oracle ≠ independent ground truth ≠ the case's expected verdict.

## v1 reference fixture (family A: finite-domain / certificate)

A closed word tree, verifier accepts only `"aa"`:

| case | candidate | expected | why |
| --- | --- | --- | --- |
| A-supported | `a` | supported | reaches accepted terminal `aa` |
| A-unsupported-subtree | `b` | unsupported | subtree fully covered, only `bb` (rejected) |
| A-unsupported-terminal | `c` | unsupported | terminal `c` rejected |
| A-unknown-incomplete | `d` | unknown | coverage partial → never pruned |

## Verification

```bash
python -m pytest tests/test_harnesses/test_solver_bench.py -q   # 6 passed
python -m scripts.run_solver_bench --describe
python -m scripts.run_solver_bench --all                        # passed, 0 false prunes
python -m ruff check src/slm_training/harnesses/solver_bench.py scripts/run_solver_bench.py
python -m scripts.repo_policy
```

Tests cover: independent ground truth matches expected on every case; the oracle
agrees with ground truth and its certificates replay; a **deliberately faulty
prune is caught** (`false_unsupported`); unknown is preserved; the manifest is
deterministic and order-independent; and the case schema rejects a bad verdict.

## Scope / deferred

Family A (finite-domain/certificate) is committed with exact independent ground
truth. Families **B–E** (cross-hole coupling & multi-pass closure, verification
capsules, topology-edit domains, opaque regions & late realization) reuse the
same harness and the merged VSS2/VSS3 owners (`capsule_solver`, `topology_solver`,
surface realizer) and are **deferred** to follow-on fixtures; the next matrix issue
(VSS4-02) can consume this benchmark as a hard correctness gate.
