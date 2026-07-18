# SLM-71 VSS3-03 topology solver decode wiring (2026-07-18)

**Kind:** fixture decode wiring / mechanism increment only.
**Claim level:** none (no ship, no quality, no promotion, no gate touched).
**Checkpoint:** none trained.
**Linear:** SLM-71 (VSS3-03).

## What changed

Implemented the disabled-by-default config and decode seam that lets
`GrammarDiffusionModel` run finite-domain exact closure over the topology edit
space before ranked expansion decisions. The default decode path is unchanged;
all new flags are off by default and old checkpoints/configs load with defaults.

This is the in-scope slice of the stale `slm-71-vss3-03-capsule-aware-exact-closure`
mega-fork reapplied onto current `main`. The canonical `dsl/solver/*` modules,
`dsl/surface.py`, the supervision corpus, and VSS3-02/04/05 already live on
`main`; only the genuinely-new topology-solver decode seam is brought here. Every
re-implementation of a file already on `main` (solver `__init__`/`capsule_solver`/
`topology_adapter` rewrites, factory rewrites, README/MODEL_CARD/pyproject/CI/
dashboard, and doc clobbers) was dropped.

## Files changed

New:
- `src/slm_training/dsl/solver/topology_solver.py` (torch-free expander/verifier + exact-closure prune)
- `tests/test_models/test_grammar_diffusion_solver.py` (torch-gated via `pytest.importorskip`)
- `docs/design/iter-slm71-topology-solver-decode-wiring-20260718.md` (this note)

Additive edits onto `main`'s current versions (default-off, no behavior change when disabled):
- `src/slm_training/models/grammar_diffusion.py` — `GrammarDiffusionConfig`
  topology-solver fields, `_topology_solver_survivors` hook, guarded prune in
  `_decode_one` (proposal tuples now carry an explicit edit action).
- `src/slm_training/dsl/solver/topology_adapter.py` — new `derive_topology_state`
  helper and `parent_id` hole-domain metadata for tree reconstruction inside the
  expander. `TopologyEdit.to_value`/`from_value` keep `main`'s canonical
  list payload (length-4) unchanged.
- `src/slm_training/dsl/solver/__init__.py` — export `derive_topology_state` and
  `TopologyNodeLike` (purely additive; `TopologyHole` export retained).
- `src/slm_training/harnesses/model_build/config.py` — `topology_solver_*`
  `ModelBuildConfig` fields.
- `src/slm_training/harnesses/model_build/factory.py` — runtime-override
  passthrough, `build_model` mapping into `GrammarDiffusionConfig`, and the
  `topology_capsule_solver` requires `topology_verified_solver` guard.

## Verification (torch-free gates local; torch tests CI-deferred)

```bash
PYTHONPATH=src python -m pytest tests/test_dsl -q
PYTHONPATH=src python -m pytest tests/test_models -q -k "not <torch-only>"
PYTHONPATH=src ruff check src/slm_training/dsl/solver src/slm_training/models/grammar_diffusion.py
PYTHONPATH=src python -m scripts.repo_policy
git diff --check
```

The torch-gated `tests/test_models/test_grammar_diffusion_solver.py` (defaults,
config round-trip, disabled-parity, enabled seam invocation, monotone prune,
invalid capsule/verified combination) skips locally because torch is not
installed; it is deferred to CI where torch is present.

## Honest caveats

- Wiring / mechanism increment only: no model was trained, nothing is shipped,
  no gate or metric was touched, meaningful-parse is out of scope.
- The full capsule-aware path (`topology_capsule_solver=True`) is only gated and
  validated; the `CapsuleProblemBuilder` / `solve_capsule_graph` plumbing into
  `GrammarDiffusionModel` is future work.
- Reversible remasking / backtracking through the search controller is not yet
  wired; the current seam filters one phase's proposals using one-pass exact
  closure.
- No model/energy ranker is implemented yet; closure survivors are filtered and
  the existing argmax proposal path still ranks within the survivors.
- No train/eval/matrix/bench run was performed, so no `MODEL_CARD.md` update is
  required.
