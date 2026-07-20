# SLM-138: Shared recursive denoiser tower fixture (slm138_recursive_denoiser_fixture)

Matrix set: `slm138-shared-recursive-denoiser`  
Version: `slm138-v1`  
Status: **wiring_only**

## What this exercises

A drop-in ``SharedRecursiveDenoiserTower`` that preserves the ``DenoiserTower`` public contract. The fixture builds tiny TwoTower models for both ``stacked`` and ``shared_recursive`` denoiser architectures, runs forward passes and training_loss, verifies shapes/gradients, confirms object-identity weight sharing across recursions, and round-trips a recursive checkpoint.

## Architectures

`stacked`, `shared_recursive`

## Forward shapes

- stacked: `[2, 6, 32]`
- recursive: `[2, 6, 32]`

## Losses

- stacked: `40.054848`
- recursive: `67.880974`

## Recursive weight sharing

- F-update distinct layer objects: 1
- G-update distinct layer objects: 1
- Total shared transition layers: 2

## Deep-supervision metrics

- `recursive_depth_loss_0`: 26.03812026977539
- `recursive_depth_loss_1`: 23.953886032104492
- `recursive_depth_supervision_loss`: 33.3280029296875

## Checkpoint round-trip

Recursive checkpoint save/load OK: **True**

## Fixture caveat

Wiring-only evidence. Full matched-block evaluation arms and GPU training are deferred.
