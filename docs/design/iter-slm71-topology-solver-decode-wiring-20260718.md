# SLM-71 VSS3-03 topology solver decode wiring (2026-07-18)

**Kind:** fixture decode wiring / replay evidence only.  
**Claim level:** none (no ship, no quality, no promotion).  
**Checkpoint:** none trained.  
**Pull request:** `slm-71-vss3-03-capsule-aware-exact-closure`.

## What changed

Implemented the disabled-by-default config and decode seam that lets
`GrammarDiffusionModel` run finite-domain exact closure over the topology edit
space before ranked expansion decisions.  The default decode path is unchanged;
all new flags are off by default and old checkpoints load with defaults.

## Files changed

- `src/slm_training/models/grammar_diffusion.py`
- `src/slm_training/dsl/solver/topology_solver.py` (new)
- `src/slm_training/dsl/solver/topology_adapter.py`
- `src/slm_training/dsl/solver/__init__.py`
- `src/slm_training/harnesses/model_build/config.py`
- `src/slm_training/harnesses/model_build/factory.py`
- `tests/test_models/test_grammar_diffusion_solver.py` (new)
- `tests/test_dsl/test_topology_adapter.py`
- `docs/design/grammar-topology-diffusion.md`

## Verification

```bash
python -m pytest tests/test_models/test_grammar_diffusion.py tests/test_models/test_grammar_diffusion_solver.py tests/test_dsl/test_topology_adapter.py tests/test_dsl/test_capsule_solver.py -q
python -m scripts.repo_policy
.githooks/check-changed
```

Result: 144 passed (full changed-check suite: 575 passed, 5 skipped) and all
policy/diff checks clean.

## Honest caveats

- The full capsule-aware path (`topology_capsule_solver=True`) is only gated
  and validated; the `CapsuleProblemBuilder` / `solve_capsule_graph` plumbing
  into `GrammarDiffusionModel` is future work.
- Reversible remasking / backtracking through the search controller is not yet
  wired; the current seam filters one phase's proposals using one-pass exact
  closure.
- No model/energy ranker is implemented yet; closure survivors are filtered and
  the existing argmax proposal path still ranks within the survivors.
- No train/eval/matrix/bench run was performed, so no `MODEL_CARD.md` update is
  required.
