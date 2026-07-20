# SLM-155 (SPV3-02): Matched AR vs plan-conditioned X22 factorization (slm155_fixture)

Matrix set: `slm155_factorization_comparison`

Version: `spv3-02-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production X22 checkpoint was loaded, and no ship-gate claim is made.

## Hypothesis

For short OpenUI semantic decision streams, direct autoregressive legal-action scoring matches or exceeds plan-conditioned X22 semantic quality at lower deployed cost; a small AR→X22 hybrid edit tail adds a useful Pareto point only when AR realization contains recoverable structural errors.

## Falsifier

X22 produces materially better semantic outcomes at comparable cost, or the hybrid duplicates the better parent at higher cost, or plan features silently alter legal candidate membership between families.

## Common config

| Key | Value |
| --- | --- |
| dsl_pack | openui |
| plan_source | predicted |
| compiler_honesty_mode | production |
| scorer_variant | mlp |
| scorer_seed | 0 |
| n_train_decisions | 64 |
| n_eval_decisions | 16 |
| x22_max_depth | 4 |
| x22_beam_width | 4 |
| equal_forward_budget | 64 |
| seeds | [0, 1, 2] |
| metric_versions | {'meaningful': '2.0.0'} |

## Arms

| Arm | Family | Promotable | Diagnostic | Description |
| --- | --- | --- | --- | --- |
| AR-G | ar | True | False | Greedy autoregressive scorer over the live legal action set. |
| AR-B | ar | True | False | AR scorer with a bounded beam/k-best calibration placeholder. |
| AR-C | ar | True | False | AR greedy followed by the shared global semantic critic/selector. |
| X-M | x22 | True | False | Canonical minimal valid X22 seed with no learned plan. |
| X-P | x22 | True | False | Plan-conditioned seed followed by conflict-slice repair. |
| X-C | x22 | True | False | X-P plus the same final global critic/selector. |
| H-1 | hybrid | True | False | AR greedy program followed by one bounded X22 refinement phase. |
| H-K | hybrid | True | False | AR program followed by the smallest calibrated K edit budget (K=2 fixture). |
| H-C | hybrid | True | False | H-1 plus the same final critic/selector. |
| gold_ar | ar | False | True | Gold-plan AR diagnostic ceiling. |
| gold_x22 | x22 | False | True | Gold-plan X22 diagnostic ceiling. |
| oracle_selector | ar | False | True | Oracle candidate/beam selector diagnostic. |

## Results

| Arm | Seed | Records | Mean semantic score | Forwards | Edits | Verifier calls |
| --- | --- | --- | --- | --- | --- | --- |
| AR-G | 0 | 16 | 1.000 | 5.8 | 0.0 | 0.0 |
| AR-G | 1 | 16 | 1.000 | 5.8 | 0.0 | 0.0 |
| AR-G | 2 | 16 | 1.000 | 5.8 | 0.0 | 0.0 |
| AR-B | 0 | 16 | 1.000 | 8.8 | 0.0 | 0.0 |
| AR-B | 1 | 16 | 1.000 | 8.8 | 0.0 | 0.0 |
| AR-B | 2 | 16 | 1.000 | 8.8 | 0.0 | 0.0 |
| AR-C | 0 | 16 | 1.000 | 6.8 | 0.0 | 0.0 |
| AR-C | 1 | 16 | 1.000 | 6.8 | 0.0 | 0.0 |
| AR-C | 2 | 16 | 1.000 | 6.8 | 0.0 | 0.0 |
| X-M | 0 | 4 | 0.000 | 64.0 | 1.0 | 16.0 |
| X-M | 1 | 4 | 0.000 | 64.0 | 1.0 | 16.0 |
| X-M | 2 | 4 | 0.000 | 64.0 | 1.0 | 16.0 |
| X-P | 0 | 4 | 0.000 | 64.0 | 1.0 | 16.0 |
| X-P | 1 | 4 | 0.000 | 64.0 | 1.0 | 16.0 |
| X-P | 2 | 4 | 0.000 | 64.0 | 1.0 | 16.0 |
| X-C | 0 | 4 | 0.000 | 65.0 | 1.0 | 16.0 |
| X-C | 1 | 4 | 0.000 | 65.0 | 1.0 | 16.0 |
| X-C | 2 | 4 | 0.000 | 65.0 | 1.0 | 16.0 |
| H-1 | 0 | 16 | 0.000 | 69.8 | 2.0 | 16.0 |
| H-1 | 1 | 16 | 0.000 | 69.8 | 2.0 | 16.0 |
| H-1 | 2 | 16 | 0.000 | 69.8 | 2.0 | 16.0 |
| H-K | 0 | 16 | 0.000 | 133.8 | 4.0 | 32.0 |
| H-K | 1 | 16 | 0.000 | 133.8 | 4.0 | 32.0 |
| H-K | 2 | 16 | 0.000 | 133.8 | 4.0 | 32.0 |
| H-C | 0 | 16 | 0.000 | 70.8 | 2.0 | 16.0 |
| H-C | 1 | 16 | 0.000 | 70.8 | 2.0 | 16.0 |
| H-C | 2 | 16 | 0.000 | 70.8 | 2.0 | 16.0 |
| gold_ar | 0 | 16 | 1.000 | 5.8 | 0.0 | 0.0 |
| gold_ar | 1 | 16 | 1.000 | 5.8 | 0.0 | 0.0 |
| gold_ar | 2 | 16 | 1.000 | 5.8 | 0.0 | 0.0 |
| gold_x22 | 0 | 4 | 1.000 | 64.0 | 1.0 | 16.0 |
| gold_x22 | 1 | 4 | 1.000 | 64.0 | 1.0 | 16.0 |
| gold_x22 | 2 | 4 | 1.000 | 64.0 | 1.0 | 16.0 |
| oracle_selector | 0 | 16 | 1.000 | 0.0 | 0.0 | 0.0 |
| oracle_selector | 1 | 16 | 1.000 | 0.0 | 0.0 | 0.0 |
| oracle_selector | 2 | 16 | 1.000 | 0.0 | 0.0 | 0.0 |

## Verdict

This is a fixture wiring run. It validates that the factorization comparison manifest is honest (gold/oracle arms non-promotable), that AR and X22 arms can be evaluated under a common trace envelope, that the hybrid AR→X22 boundary preserves lineage, and that cost accounting is deterministic. Real quality/cost claims require trained models, matched capacity, AgentV evaluation, and measured wall-clock latency.
