# VSS2-02: Capsule solve plans, SCC joint solving, and interface summaries

**Issue:** SLM-66  
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

A deterministic capsule solver coordinator in `src/slm_training/dsl/solver/capsule_solver.py`:

- `CapsuleInterfaceSummary`, `CapsuleProblem`, `CapsuleSolvePlan`, `PerCapsuleResult`, `CapsuleSolveResult`, `Disagreement`, `CapsuleCounters`, `BindingSummary`, `SlotSummary`, `ExternalInput` — immutable, JSON-safe dataclasses with `to_dict`/`from_dict`.
- `build_capsule_solve_plan(graph)` — topological stage planning that:
  - Computes dependency stages from the capsule graph.
  - Groups independent capsules that share no edges into the same stage.
  - Marks strongly-connected components (SCCs) as joint problems so they are solved together.
- `solve_capsule_graph(...)` — dependency-order solve loop that:
  - Solves each stage in order.
  - Blocks a capsule when a predecessor summary is unknown (conservative predecessor check).
  - Materializes predecessor outputs into successor inputs.
  - Records local-pass/global-fail verifier disagreements.
  - Updates counters for support queries, solver nodes, verifier calls, and joint solves.

Pack integration in `src/slm_training/dsl/pack.py`:

- Added optional slots to `DslPack`:
  - `capsule_problem_builder`
  - `capsule_summary_extractor`
  - `capsule_materializer`
  - `capsule_local_oracle`
  - `capsule_global_oracle`
- Fixed a latent registration ordering bug: `_ensure_builtin_packs()` now uses a `_BUILTINS_LOADED` flag instead of `_PACKS` truthiness, so registering a custom pack before the builtins are loaded no longer hides `openui`/`toy-layout`/`graphql`.

Serialization helpers:

- `src/slm_training/dsl/solver/controller.py` — added `to_dict`/`from_dict` to `TerminalOutcome`, `SearchDecision`, `Nogood`, and `SearchResult`.
- `src/slm_training/dsl/solver/closure.py` — added `CertifiedDeduction.from_dict`.
- `src/slm_training/dsl/solver/__init__.py` — exports the capsule solver public API.

Support oracle tweak:

- `src/slm_training/dsl/solver/support.py` — the enumerative oracle now prefers an unresolved hole when expanding a multi-hole child, while still falling back to the first hole for singleton-hole expanders.

## Verified

- `ruff check` passes.
- `python -m compileall` passes.
- `pytest tests/test_dsl/test_solver_controller.py tests/test_dsl/test_solver_closure.py tests/test_dsl/test_solver_support.py tests/test_dsl/test_solver_state.py tests/test_dsl/test_capsule_solver.py tests/test_data/test_progspec.py -q` → 110 passed.
- `.githooks/check-changed` → all checks passed (`tests/test_dsl`, `tests/test_harnesses/model_build`: 418 passed, 5 skipped, 12 deselected).
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.

## Design boundaries preserved

- The solver is coordinator wiring: it delegates per-capsule search to the existing VSS1 controller/state/closure/support stack.
- No new model, checkpoint, or ship gate is introduced.
- Conservative blocking and disagreement recording keep the contract honest: a missing predecessor summary does not silently become a pass.

## Caveats

- This is solver wiring only. Real capsule-level verification needs model-backed oracles, pack hooks for the new `DslPack` slots, and end-to-end fixture runs.
- No model, checkpoint, or ship gate is claimed.
