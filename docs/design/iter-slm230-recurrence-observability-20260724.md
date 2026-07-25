# SLM-230 recurrence observability and anytime audit

Verdict: **stagnant**

Report hash: `7e9534057fa22bd041366f62cd1ba24e02c97b3b3095b4d726601f17063a8cbc`

This is a bounded scratch-checkpoint diagnostic, not a ship, semantic, training-default, or serving-default claim.

## Recipe and evidence boundary

- Checkpoint: `outputs/runs/slm230_bounded_recursive_r4_r2/checkpoints/last.pt` (`1604b2cb9282928fa0969ecbbe7d78c9aa4b9907f74d0d58936bfc298a88b28a`)
- Train recipe: CPU scratch, 4 optimizer steps, 97 fixture-source records, trained R=4
- Calibration/final: smoke n=2 / held_out n=2
- AgentV: `{"durationMs": 21, "executionErrors": 0, "failed": 0, "meanScore": 1, "passed": 4, "total": 4}`
- Clean evidence: `True`
- No test-R extrapolation; no checkpoint sync or promotion.

## Depth-wise heldout observations

| record | depth | CE | accuracy | KL prev | JS prev | top1 stable | parse | structure | reward | block evals |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |
| held_out_form_01 | 1 | 24.253565 | 0.000000 | — | — | — | False | 0.000000 | 0.000000 | 6 |
| held_out_form_01 | 2 | 22.227669 | 0.000000 | 0.067426 | 0.012173 | False | False | 0.000000 | 0.000000 | 12 |
| held_out_form_01 | 3 | 20.947107 | 0.000000 | 0.057431 | 0.011527 | False | False | 0.000000 | 0.000000 | 18 |
| held_out_form_01 | 4 | 20.076065 | 0.000000 | 0.052160 | 0.011394 | False | False | 0.000000 | 0.000000 | 24 |
| held_out_dual_card_01 | 1 | 25.183479 | 0.000000 | — | — | — | False | 0.000000 | 0.000000 | 6 |
| held_out_dual_card_01 | 2 | 22.839157 | 0.000000 | 0.167400 | 0.024927 | False | False | 0.000000 | 0.000000 | 12 |
| held_out_dual_card_01 | 3 | 21.468895 | 0.000000 | 0.125681 | 0.026343 | False | False | 0.000000 | 0.000000 | 18 |
| held_out_dual_card_01 | 4 | 20.599167 | 0.000000 | 0.070971 | 0.016633 | False | False | 0.000000 | 0.000000 | 24 |

## Anytime policies and matched controls

| policy | mean depth | block evals | latency ms | parse | structure | reward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed:1` | 1.000000 | 6.000000 | 53.790676 | 0.000000 | 0.000000 | 0.000000 |
| `fixed:2` | 2.000000 | 12.000000 | 93.642417 | 0.000000 | 0.000000 | 0.000000 |
| `fixed:3` | 3.000000 | 18.000000 | 150.122467 | 0.000000 | 0.000000 | 0.000000 |
| `fixed:4` | 4.000000 | 24.000000 | 186.577440 | 0.000000 | 0.000000 | 0.000000 |
| `kl_plateau:4` | 4.000000 | 24.000000 | 186.577440 | 0.000000 | 0.000000 | 0.000000 |
| `oracle:4` | 1.000000 | 6.000000 | 53.790676 | 0.000000 | 0.000000 | 0.000000 |
| `topk_stable:4` | 4.000000 | 24.000000 | 186.577440 | 0.000000 | 0.000000 | 0.000000 |
| `kl_histogram_time_shuffle` | 4.000000 | 24.000000 | 186.577440 | 0.000000 | 0.000000 | 0.000000 |

Early exit qualified: **False**.

The policy cannot qualify from zero/invalid quality, and must beat both the closest fixed-average-depth control and the identical-histogram time-shuffled control.

## Exact-state and semantic limits

This bounded source has no provenance-bound DecisionEvent candidate artifact. Legal-renormalized KL, D_good/D_bad, and protected exact-state claims are therefore censored rather than filled with full-vocabulary surrogates. The SemanticFloorGateV1 verdict remains inconclusive, so strict semantic improvement is not authorized.

## Reproduction

```bash
timeout 170s env PYTHONPATH=src .venv/bin/python -m scripts.run_slm230_recurrence_observability --checkpoint outputs/runs/slm230_bounded_recursive_r4_r2/checkpoints/last.pt --test-dir src/slm_training/resources/data/eval/e763_symbol_only_eval_r2_20260722 --check
```
