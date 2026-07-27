# Lever experiment: SFT.steps 8 vs 48 (seed=42)

**Honesty:** `fixture_or_scratch` / diagnostic smoke `n=3`. **Not a ship claim.**

## Hypothesis
Increasing SFT steps from 8→48 (fixed seed=42, same `wf_smoke_v2` data, twotower/scratch/cpu, choice tokenizer) reduces `empty_prediction_count` / `decode_timeout_count` and raises `parse_rate` on the smoke suite.

## Lever
- **Name:** `SFT.steps`
- **Baseline arm:** `exp_lever_sft_steps_baseline_s8_seed42` (steps=8)
- **Treatment arm:** `exp_lever_sft_steps_treatment_s48_seed42` (requested 48; actual completed steps recorded below)

## Recipe (shared)
| Field | Value |
| --- | --- |
| seed | 42 (`recipe.seed` in train_summary) |
| model | twotower |
| context_backend | scratch |
| device | cpu |
| output_tokenizer | choice |
| train_dir | outputs/data/train/wf_smoke_v2 |
| test_dir | outputs/data/eval/wf_smoke_v2 |
| eval suite | smoke |
| eval_limit | 3 |
| grammar_constrained | True (production path; not unconstrained control) |

## Train outcomes
| Arm | stopped_on | steps completed | last_loss | wall_s |
| --- | --- | ---: | ---: | ---: |
| baseline s8 | steps | 8 | 29.6966 | 34.32 |
| treatment s48 | wall_time_budget | 36 | 12.0265 | 157.20 |

Note: treatment hit **wall_time_budget** before full 48 steps (completed 36); still a pure steps-request lever with fixed seed.

## Metrics (smoke n=3)
| Metric | baseline s8 | treatment s48 | delta |
| --- | ---: | ---: | ---: |
| `parse_rate` | 0.0 | 0.3333333333333333 | 0.3333333333333333 |
| `exact_match` | 0.0 | 0.0 | 0.0 |
| `meaningful_program_rate` | 0.0 | 0.0 | 0.0 |
| `reward_score` | 0.0 | 0.30233333333333334 | 0.30233333333333334 |
| `decode_timeout_count` | 3 | 2 | -1 |
| `empty_prediction_count` | 3 | 2 | -1 |
| `placeholder_fidelity` | 0.0 | 0.3333333333333333 | 0.3333333333333333 |
| `structural_similarity` | 0.0 | 0.02 | 0.02 |
| `latency_ms_p50` | 12007.16 | 12000.44 | -6.719999999999345 |
| `n` | 3 | 3 | 0 |

## Decision
- parse_rate **rose** 0.0 → 0.3333333333333333
- decode_timeout_count **fell** 3 → 2
- empty_prediction_count **fell** 3 → 2
- reward_score **rose** 0 → 0.30233333333333334
- meaningful_program_rate remained **0** (still not useful programs)

**Accept SFT.steps as a positive lever on this micro-recipe** (partial quality lift). Next cycle: decode-timeout / gen-steps levers on the treatment checkpoint for remaining timeouts.

Captured: 2026-07-27T13:31:49.711115+00:00
