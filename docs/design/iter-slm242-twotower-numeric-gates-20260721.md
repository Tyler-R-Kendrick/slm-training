# SLM-242 (RSC-A06): Fail-closed numeric/schedule gates for TwoTower configs

Run id: `iter_slm242_twotower_numeric_gates`
Status: **complete** (wiring/test-only; no quality claim)
Date: 2026-07-21

## What this is

Linear SLM-242 asks for a fail-closed property gate over every weight and
schedule vector used by the TwoTower training, decode, curriculum, and
auxiliary-loss surfaces. Prior issues (SLM-237/238, RSC-A01/A02) fixed
recursive-depth supervision defects, but the validation still happened at
various call sites inside `TwoTowerModel.training_loss`. This issue centralizes
and hardens the gate so that invalid numeric configs raise at construction time
with a field path, instead of silently truncating, zeroing, or ignoring a
vector.

## What changed

### New module: `src/slm_training/models/twotower_numeric_gates.py`

Typed primitive validators and two config-level entry points:

- `validate_model_build_config(cfg)` — called from `ModelBuildConfig.__post_init__`
- `validate_twotower_config(cfg)` — called from `TwoTowerConfig.__post_init__`

Primitives include: `finite_scalar`, `non_negative_scalar`, `positive_scalar`,
`interval_scalar`, `exact_length_vector`, `non_empty_vector`,
`finite_non_negative_vector`, `positive_sum_vector`,
`normalized_probability_vector`, `strictly_increasing_sequence`,
`unique_enum_sequence`, `paired_equal_length_sequences`,
`supported_capability_requirement`.

### Config construction gates

- `ModelBuildConfig.__post_init__` now calls `validate_model_build_config`.
  Rejects negative/NaN/inf weights, `mask_min`/`mask_max` outside `[0, 1]` or
  out of order, unsorted `*_stages` / `*_buckets`, and invalid recursive-depth
  weight vectors.
- `TwoTowerConfig.__post_init__` now calls `validate_twotower_config` and, when
  recursive-depth supervision is enabled, `validate_recursive_depth_supervision`.
  The SLM-237 defects are now caught when the config is built.

### CLI flags

`scripts/train_model.py` gained:

- `--recursive-depth-aux-mode {off,intermediate_only,all_depths,legacy_all_depths}`
- `--recursive-depth-aux-weight FLOAT`

Both are threaded through `ModelBuildConfig` and the model-build factory into
`TwoTowerConfig`.

### Tests

New file: `tests/test_models/test_twotower_numeric_gates.py` (28 tests):

- Primitive validator unit tests.
- `ModelBuildConfig` rejection tests for bad scalars, masks, LTR stages, and
  diffusion buckets.
- Six recursive-depth defect cases now caught at `TwoTowerConfig` construction
  time.

Updated: `tests/test_models/test_recursive_denoiser.py`. Three existing
fail-closed tests now expect the raise at model construction rather than inside
`training_loss`, reflecting the earlier gate placement.

## Verification

```text
python -m pytest -q tests/test_harnesses/model_build tests/test_models tests/test_versioning
908 passed, 1 skipped, 15 deselected

python -m pytest tests/test_models/test_twotower_numeric_gates.py \
                 tests/test_models/test_recursive_denoiser.py \
                 tests/test_harnesses/model_build/test_factory_overrides.py -q
143 passed, 1 skipped

python -m scripts.repo_policy
ok

python -m scripts.verify_version_stamps --check
ok (model.twotower v81, model.recursive_denoiser v10 no-bump)

git diff --check
ok
```

## Version bumps

- `model.twotower`: `v80 -> v81`
- `model.recursive_denoiser`: `v10` no-bump (test expectations updated only)

## Claim class

Wiring/test-only. No training run, no checkpoint, no quality or efficiency
conclusion is drawn.
