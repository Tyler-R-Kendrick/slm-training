# CAP4-04 compiler-routed block sparsity and state-family micro-experts — fixture results

**Experiment:** `CAP4-04 compiler-routed block sparsity and state-family experts fixture`  
**Artifact:** [`iter-cap4-04-block-sparsity-20260718.json`](iter-cap4-04-block-sparsity-20260718.json)  
**Run date:** 2026-07-18  
**Claim:** Wiring evidence only. No ship gate, no checkpoint promotion, no wall-clock claim.

## What changed

- Added `StateFamilyKey` and `StateFamilyRouter` in
  `src/slm_training/models/quantization/block_sparsity.py` for deterministic,
  versioned routing keys derived from `StateContext`.
- Added `BlockMaskedLinear`: a per-route block-sparse linear layer with
  whole-block masks and an all-active dense fallback when no route is supplied.
- Added `StateFamilyExpert`: a low-rank residual expert bank with a shared
  dense path; route 0 is the unknown-family fallback.
- Added `CompilerRoutedMLP`, `CompilerRoutedTransformerBlock`, and
  `CompilerRoutedDenoiserTower` to exercise the mechanisms through a mirrored
  denoiser tower.
- Extended `TensorCost` and `compute_tensor_cost` in
  `src/slm_training/models/quantization/cost.py` with optional active-numel and
  active-byte fields, plus a `compute_block_sparse_cost` helper for routed
  modules.
- Added `block_sparse_ternary_format` and `state_family_expert_format` with
  honest kernel-capability metadata (reference PyTorch path only).
- Added regression tests in `tests/test_models/test_block_sparsity.py`.
- Added fixture script `scripts/run_block_sparsity_fixture.py`.
- Registered `model.quantization` component at `v1` in
  `src/slm_training/resources/versions.json`.

## Recipe

| Field | Value |
| --- | --- |
| Device | CPU |
| Synthetic traces | 64 |
| Sequence length | 8 |
| Context length | 4 |
| State families | `card_root`, `bind_arg0`, `bind_arg1`, `literal_text` |
| `d_model` | 64 |
| `n_layers` | 2 |
| `n_heads` | 4 |
| `n_routes` | 5 (including unknown fallback) |
| `block_size` | 16 |

## Fixture results

| Variant | MLP active bytes | Active ratio | Mean L2 vs dense | Max abs diff vs dense |
| --- | --- | --- | --- | --- |
| `block_mask_50pct` | 16,916 | 94.7% | 2.88 | 0.81 |
| `state_family_expert_r8` | 217,740 | 91.3% | 36.32 | 11.35 |

The active ratios are modest on this toy model because the small `d_model`
makes the shared path dominate and the block size is coarse.  The numbers are
algorithmic active-byte estimates, not wall-clock or energy measurements.

## Verification

- `py_compile` passes for `block_sparsity.py`, `cost.py`, `formats.py`,
  `quantization/__init__.py`, the fixture script, and the regression tests.
- `pytest tests/test_models/test_block_sparsity.py` passes (17/17).
- `pytest tests/test_models/test_quantization.py` passes (21/21).
- `python -m scripts.verify_version_stamps --check` passes.
- `python -m scripts.repo_policy` passes.
- `.githooks/check-changed` passes.
- `git diff --check` passes.

## Honest caveats

- This is a **wiring fixture** on synthetic state families with randomly
  initialized models; it does not exercise the full OpenUI grammar,
  `rico_held`, or any ship gate.
- The block-mask path executes dense matrix multiplication with zeroed blocks;
  an optimized block-sparse kernel is needed for real wall-clock savings.
- The expert path loops over active routes; a fused gather/scatter kernel is
  needed for real speed-up.
- Byte costs are analytical packing estimates, not measured on-device.
- No checkpoint was written or promoted, so `docs/MODEL_CARD.md` is unchanged.
