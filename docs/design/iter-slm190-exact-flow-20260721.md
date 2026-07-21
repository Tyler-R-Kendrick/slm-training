# SLM-190 (FFE2-02): exact finite-state CTMC reference fixture (slm190-exact-flow-20260721)

Matrix set: `slm190_exact_flow`

Version: `ffe2-02-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no ship-gate claim is made.

## Hypothesis

An exact finite-state CTMC reference over compiler-certified legal edits reveals objective-dependent differences in total hazard, endpoint distribution, and path statistics, and most natural quotient partitions over program structure are not strongly lumpable.

## Falsifier

On every representative exact domain, normalized next-edit CE plus a fixed time schedule reproduces the rate-based endpoint distribution and path statistics within tolerance, and the chosen quotient partitions are strongly lumpable.

## Domains

Total domains: 3
Total cases: 12
Objective rows: 24
Lumpability tests: 24

## Cases

| case_id | domain | rate_fn | time | n_states | n_transitions | mass_error | tv_exact_vs_gillespie | multipath_entropy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| toy_layout__uniform_rate__t1.0 | toy_layout | uniform_rate | 1.0 | 217 | 240 | 5.54e-13 | 0.500 | 4.585 |
| toy_layout__distance_rate__t1.0 | toy_layout | distance_rate | 1.0 | 217 | 240 | 4.75e-13 | 0.500 | 4.585 |
| toy_layout__bridge_target_rate__t1.0 | toy_layout | bridge_target_rate | 1.0 | 217 | 240 | 0.00e+00 | 0.500 | 0.000 |
| toy_layout__doob_bridge_posterior__t1.0 | toy_layout | doob_bridge_posterior | 1.0 | 217 | 240 | 5.54e-13 | 0.500 | 4.585 |
| choice_sequence__uniform_rate__t1.0 | choice_sequence | uniform_rate | 1.0 | 83 | 82 | 6.44e-14 | 0.231 | 1.000 |
| choice_sequence__distance_rate__t1.0 | choice_sequence | distance_rate | 1.0 | 83 | 82 | 6.39e-14 | 0.733 | 1.000 |
| choice_sequence__bridge_target_rate__t1.0 | choice_sequence | bridge_target_rate | 1.0 | 83 | 82 | 3.89e-15 | 0.500 | 0.000 |
| choice_sequence__doob_bridge_posterior__t1.0 | choice_sequence | doob_bridge_posterior | 1.0 | 83 | 82 | 6.44e-14 | 0.250 | 1.000 |
| canonical_edit_graph__uniform_rate__t1.0 | canonical_edit_graph | uniform_rate | 1.0 | 226 | 285 | 5.32e-13 | 0.500 | 5.833 |
| canonical_edit_graph__distance_rate__t1.0 | canonical_edit_graph | distance_rate | 1.0 | 226 | 285 | 5.32e-13 | 0.500 | 5.833 |
| canonical_edit_graph__bridge_target_rate__t1.0 | canonical_edit_graph | bridge_target_rate | 1.0 | 226 | 285 | 0.00e+00 | 0.500 | 0.000 |
| canonical_edit_graph__doob_bridge_posterior__t1.0 | canonical_edit_graph | doob_bridge_posterior | 1.0 | 226 | 285 | 5.32e-13 | 0.500 | 5.833 |

## Lumpability

| domain | partition | status | n_blocks | n_violations |
| --- | --- | --- | --- | --- |
| toy_layout | by_terminal_class | lumpable | 217 | 0 |
| toy_layout | by_state_length | lumpable | 1 | 0 |
| toy_layout | by_terminal_class | lumpable | 217 | 0 |
| toy_layout | by_state_length | lumpable | 1 | 0 |
| toy_layout | by_terminal_class | lumpable | 217 | 0 |
| toy_layout | by_state_length | lumpable | 1 | 0 |
| toy_layout | by_terminal_class | lumpable | 217 | 0 |
| toy_layout | by_state_length | lumpable | 1 | 0 |
| choice_sequence | by_terminal_class | not_lumpable | 21 | 40 |
| choice_sequence | by_state_length | not_lumpable | 5 | 40 |
| choice_sequence | by_terminal_class | not_lumpable | 21 | 40 |
| choice_sequence | by_state_length | not_lumpable | 5 | 40 |
| choice_sequence | by_terminal_class | not_lumpable | 21 | 24 |
| choice_sequence | by_state_length | not_lumpable | 5 | 24 |
| choice_sequence | by_terminal_class | not_lumpable | 21 | 40 |
| choice_sequence | by_state_length | not_lumpable | 5 | 40 |
| canonical_edit_graph | by_terminal_class | lumpable | 1 | 0 |
| canonical_edit_graph | by_state_length | lumpable | 1 | 0 |
| canonical_edit_graph | by_terminal_class | lumpable | 1 | 0 |
| canonical_edit_graph | by_state_length | lumpable | 1 | 0 |
| canonical_edit_graph | by_terminal_class | lumpable | 1 | 0 |
| canonical_edit_graph | by_state_length | lumpable | 1 | 0 |
| canonical_edit_graph | by_terminal_class | lumpable | 1 | 0 |
| canonical_edit_graph | by_state_length | lumpable | 1 | 0 |

## Disposition

**inconclusive**

Gillespie empirical distribution differed from exact endpoint by > 0.25 TV in at least one case.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The exact CTMC reference, Gillespie sampler, objective comparisons, and lumpability tests are exercised over deterministic synthetic domains, but no trained model or decode path was run. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until trained-model flow telemetry and AgentV evaluation are available.

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.
- Domains are intentionally tiny (<= a few hundred states) so enumeration stays CPU-only.
- Distance-based and bridge rates are illustrative parameterizations; production flow objectives may differ.
- The Doob h-transform bridge posterior is computed by exact matrix exponentiation on the full state graph and is therefore a training-only oracle, not an inference-time scorer.
- Lumpability tests use coarse structural partitions; finer partitions are always trivially lumpable.

## Reproducibility

```bash
python -m scripts.run_exact_flow_fixture --mode describe
python -m scripts.run_exact_flow_fixture --mode fixture
```
