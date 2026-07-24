# SLM-243 recursive update architecture gate

Verdict: **layerscale_preferred**

Report hash: `c455b1847765c1f7957676bac2640bcc2e1548bb29c0f20c5579392d305d020d`

This is a bounded scratch architecture-repair matrix, not evidence of semantic workspace, checkpoint promotion, ship readiness, or a production default change.

## Recipe and preregistration

- Records: `smoke_hero_01, smoke_button_01, held_out_form_01, held_out_dual_card_01`
- Depths: `[1, 2, 4, 6, 8]`
- Paired seeds: `[24301, 24302, 24303]`
- Six orthogonal variants; 90 total cells; zero optimizer steps.
- Thresholds: `{"hard_nonfinite": true, "maximum_gradient_norm": 100.0, "maximum_update_ratio": 2.0, "paired_cross_entropy_tolerance": 0.25, "paired_update_ratio_fraction": 0.8, "recommendation_requires_all_three_seeds": true}`

## High-depth results

| variant | finite seeds | CE mean | max update ratio | grad norm | parse rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| current_v1 | 3/3 | 7.597369 | 1.118669 | 3.838467 | 0.000000 |
| delta_only | 3/3 | 8.725881 | 0.490605 | 8.108670 | 0.000000 |
| layerscale | 3/3 | 6.286558 | 0.000489 | 2.504227 | 0.000000 |
| gated_private | 3/3 | 6.261792 | 0.008774 | 1.597918 | 0.000000 |
| current_true_empty | 3/3 | 7.597369 | 1.118669 | 3.838467 | 0.000000 |
| layerscale_private | 3/3 | 6.286558 | 0.000489 | 2.504411 | 0.000000 |

## Disposition

- Selected variant: `layerscale`
- Maximum authorized diagnostic depth: `8`
- Allowed SLM-233 modes: `('layerscale_diagnostic',)`
- Rationale: selected repair improved high-depth update stability on all paired seeds
- Blocked claims: `('semantic_workspace', 'checkpoint_promotion', 'ship_readiness', 'production_default_change')`

Prior SLM-282/230/231/232 recurrence results remain authoritative. This matrix can authorize only a later diagnostic architecture mode; one-forward masked reconstruction is not free-running reasoning.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_slm243_recursive_update_gate --check
```
