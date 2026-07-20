# SLM-176 (P14): action-shortlist retrieve-then-rerank fixture (slm176-action-shortlist-rerank-20260720)

Matrix set: `slm176_action_shortlist_rerank`

Version: `p14-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

A deterministic description-retrieval shortlist preserves the full-set top candidate for synthetic legal action sets.

## Falsifier

The retrieval shortlist drops the full-set top-1 candidate or collapses to fallback for every non-trivial legal set.

## Scenarios

| scenario_id | legal_set_size | k | seed | query_hint |
| --- | --- | --- | --- | --- |
| size8_k4_s0 | 8 | 4 | 0 | data visualization |
| size8_k8_s0 | 8 | 8 | 0 | user input form |
| size16_k4_s0 | 16 | 4 | 0 | container layout |
| size16_k8_s0 | 16 | 8 | 0 | structural root |
| size32_k4_s0 | 32 | 4 | 0 | user input form |
| size32_k8_s0 | 32 | 8 | 0 | container layout |

## Results

| scenario_id | shortlist_size | top1_retained | top5_retained | fallback | recall@k | wall_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| size8_k4_s0 | 4 | True | True | - | 1.000 | 0.002 |
| size8_k8_s0 | 8 | True | True | - | 1.000 | 0.001 |
| size16_k4_s0 | 4 | True | True | - | 1.000 | 0.001 |
| size16_k8_s0 | 8 | True | True | - | 1.000 | 0.001 |
| size32_k4_s0 | 4 | True | True | - | 1.000 | 0.002 |
| size32_k8_s0 | 8 | True | True | - | 1.000 | 0.002 |

## Aggregate

- mean recall@k: **1.000**
- mean full-set top-1 retained: **1.000**

## Disposition

**shortlist_wiring_ok**

Deterministic description retrieval retains the full-set top candidate and achieves reasonable recall@k on synthetic legal sets.  Wiring is ready for a trained-model test.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The retrieval, shortlist, and rerank plumbing are exercised over a deterministic synthetic encoder and catalog, but no real model was trained or evaluated. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained model and AgentV evaluation are available.

## Honest caveats

- The FixtureDescriptionEncoder is a deterministic hash surrogate, not a   trained language model; geometry may differ with real text encoders.
- Synthetic legal sets are random permutations, not live compiler output.
- No ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode plan-only
python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode fixture
```
