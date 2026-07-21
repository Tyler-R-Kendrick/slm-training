# SLM-138: Shared recursive denoiser tower fixture (slm138_recursive_denoiser_fixture)

Matrix set: `slm138-shared-recursive-denoiser`  
Version: `slm138-v1`  
Status: **wiring_only**

## What this exercises

A drop-in ``SharedRecursiveDenoiserTower`` that preserves the ``DenoiserTower`` public contract. The fixture builds tiny TwoTower models for both ``stacked`` and ``shared_recursive`` denoiser architectures, runs forward passes and training_loss, verifies shapes/gradients, confirms object-identity weight sharing across recursions, and round-trips a recursive checkpoint. SLM-239 (RSC-A03): RNG usage now follows an explicit, disjoint namespace contract -- see the ``rng_contract`` field below. SLM-240 (RSC-A04): the retracted same-parameter-count/layer-names claim is replaced by a real, measured ``ArchitectureComparisonReportV1`` -- see 'Architecture comparison' below.

## Architectures

`stacked`, `shared_recursive`

## Forward shapes

- stacked: `[2, 6, 32]`
- recursive: `[2, 6, 32]`

## Architecture comparison (SLM-240 / RSC-A04)

Independently measured comparison dimensions -- never a single collapsed `parity` claim (see `ArchitectureComparisonReportV1`).

- interface-compatible: **true**
- output-shape-compatible: **true**
- parameter-matched: **false**
- parameter delta: `+9248` (+14.23%) -- reproduced from `recursive_zstate_parameter_delta(d_model=32, max_len=256)`, exactly `z_latent` + `ctx_proj`
- parameter_count_denoiser (transition layers only, architecture-independent): `{'stacked': 33792, 'recursive': 33792}`
- behavioral parity: **not claimed**
- claim class: **wiring**
- block evaluations per forward: `{'stacked': 2, 'recursive': 4}`

## SLM-241 (RSC-A05) control arm table

Real, measured resource accounting per built control arm -- never a raw loss or a winner (see `docs/design/iter-rsc-a05-*` for the full formulas/residuals). Built arms: A, B, C, D, F, G. Deferred: E, H.

| arm | denoiser_arch | z_state_mode | params (Δ vs A) | block evals | matched? |
| --- | --- | --- | --- | --- | --- |
| A | `stacked` | `None` | 43040 (+0) | 2 | True |
| B | `shared_recursive` | `full` | 52288 (+9248) | 4 | False |
| C | `shared_recursive_y_only` | `y_only` | 43040 (+0) | 4 | True |
| D | `shared_recursive_no_extra_capacity` | `parameter_free` | 43040 (+0) | 4 | True |
| F | `stacked_depth_matched` | `None` | 76832 (+33792) | 4 | False |
| G | `shared_recursive` | `full` | 52288 (+9248) | 2 | False |

## Arm F dual view (block-evaluation-matched vs parameter-nearest)

Arm F (unshared depth-matched tower) has exactly one free dial (`n_layers`), so it cannot match both arm B's block-evaluation count and its parameter count simultaneously -- both real, measured constructions are reported below with an explicit residual on whichever dimension is not matched.

- Target arm: **B** -- `52288` parameters, `4` block evaluations per forward.
- Per-layer parameter cost (measured from real 1-layer/2-layer towers, never hard-coded): `16896` per layer, `9248` common (non-block) parameters.

| view | n_layers | block evals | Δ block evals vs B | params | Δ params vs B |
| --- | --- | --- | --- | --- | --- |
| block_evaluation_matched | 4 | 4 | 0 | 76832 | +24544 |
| parameter_nearest | 3 | 3 | -1 | 59936 | +7648 |

Neither row is a 'matched' claim on both dimensions at once -- `block_evaluation_matched` is the `control_arm_table` "F" row above; `parameter_nearest` is a separate construction reported only here.

## Losses

**Objective-decomposition warning:** the raw scalar losses below are *not* a quality/parameter-matched comparison -- the two architectures have different parameter counts (see 'Architecture comparison' above) and the recursive arm's loss includes deep-supervision terms whose exact weighting/mode is governed by SLM-238 (RSC-A02)'s `recursive_depth_aux_mode` (see `deep_supervision_metrics` below and `docs/design/iter-rsc-a02-*`); placing these two numbers side by side never implies one architecture is better.

- stacked: `49.943035`
- recursive: `77.007797`

## Post-update verification (restored corruption-RNG checkpoint)

- stacked: `45.294647`
- recursive: `71.959007`

## Recursive weight sharing

- F-update distinct layer objects: 1
- G-update distinct layer objects: 1
- Total shared transition layers: 2

## Deep-supervision metrics

- `recursive_depth_supervision_enabled`: True
- `recursive_depth_aux_mode`: legacy_all_depths
- `recursive_depth_aux_weight`: 1.0
- `recursive_depth_supervision_weight_sum`: 1.5
- `recursive_depth_loss_0`: 35.52149200439453
- `recursive_depth_weight_0`: 0.5
- `recursive_depth_weighted_contribution_0`: 11.840497970581055
- `recursive_depth_loss_1`: 32.49333572387695
- `recursive_depth_weight_1`: 1.0
- `recursive_depth_weighted_contribution_1`: 21.66222381591797
- `recursive_depth_supervision_loss`: 33.502723693847656

## RNG contract

- Contract version: `FixtureRngContractV1`
- Base seed: `0`
- Probe order: `stacked_first`
- Training-corruption seed: `30000`
- Namespace seeds: `{'arch_specific:shared_recursive': 70000, 'arch_specific:shared_recursive_no_extra_capacity': 90000, 'arch_specific:shared_recursive_y_only': 80000, 'arch_specific:stacked': 60000, 'arch_specific:stacked_depth_matched': 100000, 'control_only': 50000, 'model_initialization': 0, 'shape_probe_context': 20000, 'shape_probe_inputs': 10000, 'training_batch_order': 40000, 'training_corruption': 30000}`

## Clean-tree evidence gate

- Comparable/claim-grade: **True**
- code_dirty: `False`
- diff_hash: `None`

## Checkpoint round-trip

Recursive checkpoint save/load OK: **True**

## Fixture caveat

Wiring-only evidence. Full matched-block evaluation arms and GPU training are deferred.
