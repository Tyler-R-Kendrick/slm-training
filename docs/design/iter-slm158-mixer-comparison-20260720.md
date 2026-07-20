# SLM-158 (SPV3-05): Sequence-mixer comparison fixture (slm158_fixture)

Matrix set: `slm158_mixer_comparison`

Version: `spv3-05-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

## Hypothesis

A narrow sequence-mixer protocol with simplified reference implementations can expose whether non-Transformer mixers preserve task accuracy and improve latency/memory on a shared synthetic sequence task, before productionizing any one family.

## Falsifier

All mixers perform identically on the actual workload distribution, recurrent/SSM mixers require unrealistic lengths to show a cost win, or simplified references cannot be trained stably enough to separate mixer family effects from implementation noise.

## Arms

| Arm | Family | Promotable | Reference | Description |
| --- | --- | --- | --- | --- |
| T0_no_mixer | no_mixer | True | True | Mean-pooled token embedding floor with no learned sequence mixer. |
| T1_transformer | transformer | True | True | Small Transformer encoder (baseline). |
| S1_mamba_reference | mamba_reference | True | True | Simplified Mamba-family selective SSM reference. |
| L1_gated_delta_net | gated_delta_net | True | True | Simplified Gated DeltaNet-style linear-attention reference. |
| R1_rwkv_reference | rwkv | True | True | Simplified RWKV-style recurrent time-mixing reference. |
| R2_xlstm_reference | xlstm | True | True | Simplified xLSTM mLSTM-style matrix-memory reference. |
| C1_hyena_reference | hyena | True | True | Simplified Hyena-style long-convolution reference. |

## Results

| Arm | Seed | Records | Loss | Accuracy | Latency ms | Params |
| --- | --- | --- | --- | --- | --- | --- |
| T0_no_mixer | 0 | 8 | 2.287 | 0.125 | 11.232 | 4552 |
| T1_transformer | 0 | 8 | 2.482 | 0.125 | 344.205 | 30760 |
| S1_mamba_reference | 0 | 8 | 2.449 | 0.125 | 54.617 | 5625 |
| L1_gated_delta_net | 0 | 8 | 2.296 | 0.125 | 364.176 | 7689 |
| R1_rwkv_reference | 0 | 8 | 2.557 | 0.250 | 68.651 | 8712 |
| R2_xlstm_reference | 0 | 8 | 2.288 | 0.125 | 324.496 | 6666 |
| C1_hyena_reference | 0 | 8 | 2.350 | 0.125 | 74.819 | 6616 |

## Verdict

This is a fixture wiring run. The simplified reference mixers share a common input/output contract and are evaluated on a synthetic token-pattern task. Real claims require optimized kernels, the actual legal-action scorer and compiler state, measured wall-clock on target hardware, and held-out causal evaluation.
