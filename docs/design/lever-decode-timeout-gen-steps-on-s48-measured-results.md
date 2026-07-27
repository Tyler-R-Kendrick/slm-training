# Lever experiment: decode_timeout / gen_steps on s48 checkpoint

**Honesty:** `fixture_or_scratch` / diagnostic smoke `n=3`. **Not a ship claim.**

## Hypothesis
On the fixed treatment checkpoint from the SFT.steps experiment, increasing `decode_timeout_seconds` (12→30) or `gen_steps` (8→16) reduces remaining empty predictions / timeouts without retraining.

## Shared fixed inputs
| Field | Value |
| --- | --- |
| checkpoint | `outputs/runs/exp_lever_sft_steps_treatment_s48_seed42/checkpoints/last.pt` |
| seed (train) | 42 |
| eval suite | smoke n=3 |
| model | twotower |
| grammar_constrained | True |

## Arms
| Arm | run_id | lever value |
| --- | --- | --- |
| baseline | `exp_lever_sft_steps_treatment_s48_seed42` | default decode timeout / gen_steps from prior eval |
| decode_timeout_30s | `exp_lever_decode_timeout_30s_from_s48_seed42` | decode_timeout_seconds=30 |
| gen_steps_16 | `exp_lever_gen_steps_16_from_s48_seed42` | gen_steps=16 |

## Metrics
| Metric | baseline | timeout30 | gen16 |
| --- | ---: | ---: | ---: |
| `parse_rate` | 0.3333333333333333 | 1.0 | 0.3333333333333333 |
| `exact_match` | 0.0 | 0.0 | 0.0 |
| `meaningful_program_rate` | 0.0 | 0.3333333333333333 | 0.0 |
| `reward_score` | 0.30233333333333334 | 0.8436666666666667 | 0.30233333333333334 |
| `decode_timeout_count` | 2 | 0 | 2 |
| `empty_prediction_count` | 2 | 0 | 2 |
| `placeholder_fidelity` | 0.3333333333333333 | 0.7222222222222222 | 0.3333333333333333 |
| `structural_similarity` | 0.02 | 0.31583333333333335 | 0.02 |
| `latency_ms_p50` | 12000.44 | 19192.26 | 12003.23 |
| `n` | 3 | 3 | 3 |

## Decision
- **decode_timeout_30s**: better=True; deltas={'parse_rate': 0.6666666666666667, 'exact_match': 0.0, 'meaningful_program_rate': 0.3333333333333333, 'reward_score': 0.5413333333333333, 'decode_timeout_count': -2, 'empty_prediction_count': -2, 'placeholder_fidelity': 0.3888888888888889, 'structural_similarity': 0.29583333333333334, 'latency_ms_p50': 7191.819999999998, 'n': 0}
- **gen_steps_16**: better=False; deltas={'parse_rate': 0.0, 'exact_match': 0.0, 'meaningful_program_rate': 0.0, 'reward_score': 0.0, 'decode_timeout_count': 0, 'empty_prediction_count': 0, 'placeholder_fidelity': 0.0, 'structural_similarity': 0.0, 'latency_ms_p50': 2.789999999999054, 'n': 0}

Captured: 2026-07-27T13:33:02.512271+00:00
