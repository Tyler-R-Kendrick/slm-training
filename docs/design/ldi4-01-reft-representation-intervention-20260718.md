# LDI4-01 — Low-rank representation interventions (wiring memo)

**Issue:** SLM-134 (LDI4-01). **Date:** 2026-07-18. **Status:** intervention module
+ controls + artifact lifecycle, implemented and tested. **No model training or
matched matrix run** — this memo is wiring evidence; representation-space
superiority is **not** claimed without matched trainable evidence.

## What this delivers

A representation-space actuator for TwoTower that tests *where the correction
lives*, reusing the shared action evidence / objectives (no new schema, objective
library, or eval pipeline):

- `src/slm_training/models/reft_intervention_spec.py` — torch-free versioned
  `InterventionSpec` (artifact identity / fail-closed config) and the matched
  `matched_arm_specs` arm-spec builder, importable without torch (mirrors the
  `models/adapters/spec.py` split so a CLI / campaign harness can enumerate arms).
- `src/slm_training/models/reft_intervention.py` — the torch-dependent
  intervention modules (`LowRankReft`, `DiffMeanIntervention`, parent identity),
  `build_intervention`, `diffmean_vector`, and the artifact save/load lifecycle.
  Re-exports the spec symbols so the public import surface is a single module.
- `tests/test_models/test_reft_intervention.py` — 8 tests, torch-gated via
  `pytest.importorskip("torch")` (run in CI where torch is installed).

## Intervention

LoReFT-style low-rank affine subspace edit at one declared site/position, applied
to hidden state `h`:

```
h' = h + scale * (W h + b - R h) Rᵀ        # R: [rank, hidden] orthonormal rows
```

Initialising `W = R`, `b = 0` makes the edit exactly zero, so an **untrained or
disabled intervention is bit-identical to the parent**. Classification: **Adapted**
(not a faithful ReFT reproduction).

Controls (matched arm set via `matched_arm_specs`):

| Arm | Method | Trainable |
| --- | --- | --- |
| R0 parent | `no_intervention` | identity, 0 params |
| R2 DiffMean | `diffmean_fixed` | fixed train-only difference-in-means vector, 0 params |
| R3 | `reft_r1` (rank 1) | R, W, b |
| R4 | `reft_low_rank` (ranks 2/4/8) | R, W, b |

Only `R, W, b` (or nothing, for DiffMean/parent) receive gradients; the base model
stays frozen.

## Guarantees (tested)

- `no_intervention` and an untrained ReFT are **bit-identical / within 1e-6** of the
  parent.
- **`torch.autograd.gradcheck` passes** for the ReFT forward after perturbation.
- Only `R/W/b` are trainable; DiffMean and parent expose no trainable params.
- `diffmean_vector` is `mean(positive) − mean(negative)` over train-group
  activations only, applied as `h + scale·v`.
- Artifact **save/load reproduces intervention-enabled outputs** (ReFT and DiffMean).
- Config **fails closed**: unknown method, `reft_r1` with rank≠1, rank outside
  `[1, hidden]`, negative scale, or DiffMean without a vector all raise; the config
  fingerprint is part of artifact identity.

## Commands

```bash
# torch present (CI): 8 passed; torch absent: 8 skipped (importorskip)
python -m pytest tests/test_models/test_reft_intervention.py -q
python -m ruff check src/slm_training/models tests/test_models
python -m compileall -q src/slm_training/models/reft_intervention_spec.py \
  src/slm_training/models/reft_intervention.py
python -m scripts.repo_policy
```

## Scope

The intervention module, controls, config, artifact save/load, and the matched
arm-spec builder. **Deferred to a GPU run** (and gated on SLM-126's authorization —
if it ended `no_safe_direction`, only wiring + no-update diagnostics may run, and no
superiority may be claimed): the site/position hook integration into
`twotower.py`, training the R0–R5 arms under the shared local objective and
held-out guards, and the five-suite / AgentV evaluation. No training runs here.
