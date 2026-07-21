# SLM-223 (NCS0-03): SemanticFloorGateV1 fixture (slm223-semantic-floor-gate-20260721)

**Matrix set:** `slm223_semantic_floor_gate`
**Version:** `ncs0-03-v1`
**Status:** fixture
**Claim class:** wiring
**Floor threshold:** parse_rate < 0.5
**Atlas hash:** `6d8414399256736c...`
**Disposition:** no_signal — LOFO balanced accuracy 0.500 does not clear the permutation-null mean 0.500 by the required 0.15 margin (margin=0.000).

## Hypothesis

A role-weighted aggregate of SpectralAtlasV1 alpha_z, calibrated on out-of-fold training families only (direction + median threshold), can flag checkpoints under a parse-rate floor with a leave-one-family-out balanced accuracy that exceeds a label-permuted control.

## Falsifier

The calibrated gate's leave-one-family-out balanced accuracy does not exceed the label-permutation-null mean by at least 0.15, or there are too few families/runs to calibrate out-of-fold at all.

## Honest caveats

- Fixture/wiring evidence only: no trained model, checkpoint promotion, or GPU run.
- Built entirely on SLM-215 SpectralAtlasV1 rows (real or synthetic); no new spectral measurement is taken here.
- This is a diagnostic pre-screen candidate, not a promotion or ship gate. It does not replace full suite evaluation and makes no readiness claim.
- Role weights and thresholds are learned only from training-fold data inside leave-one-family-out; the tiny fixture size (2-4 runs per fold) limits statistical power and the calibration is expected to be noisy.
- The floor threshold on parse_rate is a fixed configuration knob, not fit from data; changing it changes which checkpoints count as floor-risk.
- No causal conclusion is drawn; a positive result only means the alpha_z signal correlates with the floor label in this fixture, not that it causes it.

## Summary

- Runs: 4
- Families: 2
- LOFO real balanced accuracy: 0.500
- Permutation-null mean: 0.5
- Permutation draws evaluated: 20

## Per-run gate decisions (leave-one-family-out)

| run_id | family | fold | mean α z | weighted α z | parse_rate | floor_label | gate_flag | correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixture_run_0 | family_0 | held_out_family_0 | 0.158 | 0.158 | 0.411 | True | True | True |
| fixture_run_1 | family_1 | held_out_family_1 | 0.042 | 0.042 | 0.418 | True | False | False |
| fixture_run_2 | family_0 | held_out_family_0 | 0.326 | 0.326 | 0.419 | True | True | True |
| fixture_run_3 | family_1 | held_out_family_1 | 0.136 | 0.136 | 0.420 | True | False | False |

## Full-data role weights (reported only, not used for disposition)

| role | weight (Pearson α z vs parse) |
| --- | --- |
| mlp_in | 0.270 |
| mlp_out | -0.506 |
| self_attn_k | 0.452 |
| self_attn_out | -0.133 |
| self_attn_q | 0.213 |
| self_attn_v | -0.070 |

## No-go for promotion

This report is wiring/fixture evidence only. No checkpoint promotion, GPU train, or ship gate is claimed. `SemanticFloorGateV1` is a diagnostic pre-screen candidate; it does not replace full suite evaluation.
