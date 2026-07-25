# SLM-362 (DSH1-10): matched CAP0 curriculum experiment

**Claim class:** fixture (fixture n cannot issue CERT_CAP0)

**Certificate:** rejected (prediction_identical, copy_dominated, structurally_regressive, underpowered)

## Preregistration (locked before results)

- Arms: BASE, ATOMIC_ONLY, STAGED_NO_PREFERENCE, STAGED_FULL
- Seeds: [0, 1]
- Records per arm (matched target/compute exposure): 16
- Steps x batch x lr: 8 x 4 x 0.003
- Preregistered minimum effect: paired effect > 0.0, McNemar p <= 0.1
- Stop rules: prediction_identical, copy_dominated, structurally_regressive, contaminated, underpowered
- Preregistration sha256: `52119d803914ae6b897faaec8c9ed6bed59417180f8810bf0ff1a2b8e235e7ce`

## Exposure accounting

| Arm | Records | Target chars | Families missing |
| --- | --- | --- | --- |
| ATOMIC_ONLY | 16 | 553 | [] |
| BASE | 16 | 1612 | [] |
| STAGED_FULL | 16 | 2144 | [] |
| STAGED_NO_PREFERENCE | 16 | 1777 | [] |

- Equal record count: True
- Equal source-family exposure: True

## Arm metrics (primary seed)

| Arm | canon-equiv | accepted-mass | prod-cov | structure | binding | empty | diversity | identity-base | gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| BASE | 0.227 | 0.000 | 0.200 | 1.000 | 0.000 | 0.000 | 0.121 | 0.000 | insufficient_n |
| ATOMIC_ONLY | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.030 | 0.000 | insufficient_n |
| STAGED_NO_PREFERENCE | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.030 | 0.000 | insufficient_n |
| STAGED_FULL | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.030 | 0.000 | insufficient_n |

## Paired evidence (STAGED_FULL vs controls, pooled seeds)

| Comparison | n | control | candidate | effect | p | meets effect |
| --- | --- | --- | --- | --- | --- | --- |
| compositional_held_out:STAGED_FULL_vs_BASE | 6 | 0.000 | 0.000 | 0.000 | 1.000 | False |
| compositional_held_out:STAGED_FULL_vs_ATOMIC_ONLY | 6 | 0.000 | 0.000 | 0.000 | 1.000 | False |
| prefix_completion:STAGED_FULL_vs_BASE | 12 | 0.000 | 0.000 | 0.000 | 1.000 | False |
| prefix_completion:STAGED_FULL_vs_ATOMIC_ONLY | 12 | 0.000 | 0.000 | 0.000 | 1.000 | False |

## Baselines (must fail the gate)

- All collapse baselines fail: True

## Certificate decision

- Status: **rejected**; certificate: None
- Rejection codes: ['prediction_identical', 'copy_dominated', 'structurally_regressive', 'underpowered']
- Contamination findings: []
- CAP1/CAP2 training: CLOSED (stop rule)

## Exact command

```bash
python -m scripts.run_slm362_cap0_experiment --mode aggregate
```
