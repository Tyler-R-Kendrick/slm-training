# Lever experiment: ASAP decode + continued SFT (seed=42)

**Honesty:** fixture_or_scratch / smoke n=3. **Not ship.**

## Baseline champion
`exp_lever_decode_timeout_30s_from_s48_seed42` (s48 ckpt + decode_timeout=30)

## Arms
| Arm | Lever | run_id |
| --- | --- | --- |
| ASAP | constraint-debt-routing-mode=fixed_asap, timeout=30 | exp_lever_asap_decode_s48_t30_seed42 |
| Continue SFT | initialize-from s48 + 24 steps, eval timeout=30 | exp_lever_sft_continue_s48_plus24_seed42_eval_t30 |

## Metrics
| Metric | champion | ASAP | continue+24 |
| --- | ---: | ---: | ---: |
| parse_rate | 1.0 | 1.0 | 0.0 |
| meaningful_program_rate | 0.3333333333333333 | 0.3333333333333333 | 0.0 |
| reward_score | 0.8436666666666667 | 0.8436666666666667 | 0.0 |
| decode_timeout_count | 0 | 0 | 3 |
| empty_prediction_count | 0 | 0 | 3 |
| latency_ms_p50 | 19192.26 | 10306.88 | 30001.84 |
| last_loss / steps | n/a | n/a | 8.189298629760742 / 24 (steps) |

## Decision
- ASAP vs champion: latency_better=True, quality_ok=True
- Continue SFT vs champion: better=False
- Overall campaign champion after cycle4: **asap_s48_t30**

Grammar constrained retained on all arms.

Captured: 2026-07-27T13:43:18.110816+00:00
