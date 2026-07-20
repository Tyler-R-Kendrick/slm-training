# SLM-136 / LDI4-02: SAE decision-state diagnostic fixture (ldi4_02_plan)

Matrix set: `ldi4-02-sae-decision-state`
Version: `ldi4-02-v1`
Status: **plan_only**
Site: `denoiser.block.0.residual`

## What this measures

A sparse-autoencoder diagnostic track compared against matched direct baselines (DiffMean, linear probe, ReFT-r1, direct adapter) on synthetic decision-state activations. The fixture is wiring-only: no real checkpoint, no activation capture, and no steering or interpretability claim.

## Arms

| Arm | Method | Selection | Target effect | Preservation damage | Wrong-site effect | Classification |
| --- | --- | --- | --- | --- | --- | --- |
| S0 | no_intervention | none | — | — | — | — |
| S1 | random_normalized_direction | none | — | — | — | — |
| S2 | raw_diffmean | train_only | — | — | — | — |
| S3 | linear_probe_direction | train_only | — | — | — | — |
| S4 | reft_r1 | train_only | — | — | — | — |
| S5 | direct_weight_adapter | train_only | — | — | — | — |
| S6 | top_sae_feature | train_only | — | — | — | — |
| S7 | sparse_sae_feature_set | train_only | — | — | — | — |

## Interpretation

* ``diagnostic_only`` — the arm moves the target but is dominated by direct baselines.
* ``causal_but_inferior`` — causal effect exists but is weaker than matched controls.
* ``competitive`` — localized, within preservation budget, and not beaten by controls.
* ``rejected`` — non-localized, damaging to preservation states, or otherwise unsafe.

## Fixture caveat

plan-only: no model or activation capture executed
