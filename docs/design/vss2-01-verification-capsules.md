# VSS2-01: Dependency-closed verification capsules

**Issue:** SLM-65  
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

A deterministic derived graph over existing `ScopeContract`s in `src/slm_training/data/progspec/capsules.py`:

- `DependencyKind` enum (`DEFINES`, `REFERENCE`, `CONTAINMENT`, `ROOT_OUTPUT`, `EFFECT`, `EXTERNAL`).
- Immutable JSON-safe `ScopeNode`, `ScopeEdge`, `VerificationCapsule`, and `CapsuleGraph` dataclasses.
- `derive_capsule_graph(spec: ProgramSpec) -> CapsuleGraph`:
  1. Reuses `derive_scope_contracts` for stable statement/non-statement scopes.
  2. Builds graph nodes only from `ScopeKind.STATEMENT` contracts plus a synthetic root interface node.
  3. Attaches nested `COMPONENT_CALL` / `CHILD_LIST` contracts as member paths on the nearest containing statement/root.
  4. Walks the typed AST to add `REFERENCE` edges for binder uses, `EXTERNAL` edges for slot/template inputs, and a `ROOT_OUTPUT` edge.
  5. Raises `ValueError` for forward references / undefined binders.
  6. Runs Tarjan SCC to produce dependency-closed verification capsules.

Exports added to `src/slm_training/data/progspec/__init__.py`.

## Verified

- `ruff check` passes.
- `python -m compileall` passes.
- `pytest tests/test_data/test_progspec.py` passes (17 tests, including 7 new capsule tests).
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.

## Design boundaries preserved

- `ScopeSlice` and `extract_scope_slices` remain syntax-oriented and untouched.
- `ScopeContract` / `derive_scope_contracts` remain the stable AST contract; capsules are a separate projection.
- No third scope extractor was created.
- Forward references fail closed; the graph does not weaken legality.

## Caveats

- This is fixture wiring only. The V1 graph boundary is implemented; richer effect/ containment edge semantics and capsule-level solving are follow-ups.
- No model, checkpoint, or ship gate is claimed.
