# SLM-368 (DSH2-08): matched CAP1 grounding experiment

**Claim class:** fixture (fixture n cannot issue CERT_CAP1)

**Certificate:** rejected (underpowered, prediction_identical, ignores_schema, fails_hard_contrasts, cap0_regression)

## Preregistration (locked before results)

- Arms: NL_NO_SCHEMA, SCHEMA_UNFILTERED, SCHEMA_FILTERED_SINGLE, SCHEMA_FILTERED_MULTI
- Seeds: [0, 1]
- Records per arm (matched target/compute exposure): 16
- Steps x batch x lr: 8 x 4 x 0.003
- Held-out strata: original, paraphrase, marker_permutation, counterfactual
- Preregistered minimum effect: paired effect > 0.0, McNemar p <= 0.1, paired n >= 64
- Stop rules: ignores_schema, fails_hard_contrasts, cap0_regression, underpowered, contaminated, prediction_identical
- Preregistration sha256: `5a747a73e3985393a426da8c84a0a44d313356422a7fd9ad49eeeaa9fbebb951`

## Exposure accounting

| Arm | Records | Target chars | Families missing |
| --- | --- | --- | --- |
| NL_NO_SCHEMA | 16 | 1629 | [] |
| SCHEMA_FILTERED_MULTI | 16 | 1508 | [] |
| SCHEMA_FILTERED_SINGLE | 16 | 1508 | [] |
| SCHEMA_UNFILTERED | 16 | 1629 | [] |

- Equal record count: True
- Equal source-family exposure: True
- Symbol-only targets: True (violations: [])

## Arm metrics (primary seed)

| Arm | canon-acc | facts-acc | prompt-inv | rel-sens | inv-rate | cap0-ret | gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NL_NO_SCHEMA | 0.000 | 0.000 | 0.000 | 0.083 | 1.000 | 0.000 | fail |
| SCHEMA_UNFILTERED | 0.000 | 0.000 | 0.000 | 0.167 | 1.000 | 0.000 | fail |
| SCHEMA_FILTERED_SINGLE | 0.000 | 0.000 | 0.000 | 0.167 | 1.000 | 0.000 | fail |
| SCHEMA_FILTERED_MULTI | 0.000 | 0.000 | 0.000 | 0.167 | 1.000 | 0.000 | fail |

## Paired evidence (filtered arms vs controls, pooled seeds)

| Comparison | n | control | candidate | effect | p | meets effect |
| --- | --- | --- | --- | --- | --- | --- |
| SCHEMA_FILTERED_SINGLE:vs_NL_NO_SCHEMA | 32 | 0.000 | 0.000 | 0.000 | 1.000 | False |
| SCHEMA_FILTERED_SINGLE:vs_SCHEMA_UNFILTERED | 32 | 0.000 | 0.000 | 0.000 | 1.000 | False |
| SCHEMA_FILTERED_MULTI:vs_NL_NO_SCHEMA | 32 | 0.000 | 0.000 | 0.000 | 1.000 | False |
| SCHEMA_FILTERED_MULTI:vs_SCHEMA_UNFILTERED | 32 | 0.000 | 0.000 | 0.000 | 1.000 | False |

## Schema replay probe (causal consumption, primary seed)

| Arm | relevant sensitivity | invariant rate | cap0 retention |
| --- | --- | --- | --- |
| SCHEMA_FILTERED_MULTI | 0.167 | 1.000 | 0.000 |
| SCHEMA_FILTERED_SINGLE | 0.167 | 1.000 | 0.000 |

## Certificate decision

- Status: **rejected**; certificate: None
- Rejection codes: ['underpowered', 'prediction_identical', 'ignores_schema', 'fails_hard_contrasts', 'cap0_regression']
- Contamination findings: []
- Natural-language CAP2: CLOSED (stop rule)

## Fixture limitations (honest scope)

- The on-branch tree-edit trainer fragment-validates targets through the OpenUI grammar, so mini-flow records contribute zero gradient; mini-pack suite rows measure an OpenUI-only-trained fixture model. This is recorded, not hidden, and the certificate decision above already rejects at fixture n.
- Target-char totals differ across arms because the filtered arms carry fewer unique prompt styles before matched-count replicate padding; record count and source-family coverage are exactly matched.

## Exact command

```bash
python -m scripts.run_slm368_cap1_experiment --mode aggregate
```
