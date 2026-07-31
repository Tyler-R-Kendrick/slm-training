# Semantic-factor frontier measured results (SFF-anti-e237-v1)

**Claim class:** `fixture` (synthetic ranked suite — **not** held_out promotion)  
**Campaign:** `SFF-anti-e237-v1` / `semantic_factor_frontier_measured`  
**JSON:** [semantic-factor-frontier-results.json](semantic-factor-frontier-results.json)  
**Design:** [semantic-factor-frontier.md](semantic-factor-frontier.md)

## External config bind

Metrics inventory, formulas, scorer parameters, and campaign metadata are
**external versioned JSON** (not harness constants). This run records:

| Resource | Schema | Path |
| --- | --- | --- |
| metrics | `sff_metrics/v1` | `resources/experiments/semantic_factor_frontier/metrics.v1.json` |
| scorer_params | `sff_scorer_params/v1` | `…/scorer_params.v1.json` |
| campaign | `sff_campaign/v1` | `…/campaign.v1.json` |

Exact content digests are in the JSON `config_resources.*.sha256`. Adding a
metric of an existing formula type or changing α / role weights is a resource
edit only — harness code stays put.

## Suite

| Quantity | Value |
| --- | ---: |
| Examples | 28 (26 ranked/adversarial + singleton + incomplete) |
| Seeds | 0, 1, 2 |
| Ranked rows / arm | **78** |
| Control gold accuracy | **0.00** (baseline argmax ≠ gold by design) |
| Timer | `time.perf_counter` per-arm full decision path |
| Control wall_ms mean | ~0.057 ms |

Decision path timed **independently per arm**:
`project_program_factors` + `build_factor_representation` + `score` + `select`.

## Quality × runtime (first-class)

Runtime is a **primary decision axis**, not a footnote. Slightly lower semantic
accuracy can win if wall time is ~10–100× lower. Numbers from the latest JSON
refresh (machine-local absolute ms; ratios are the stable signal).

| Arm | acc | MRR | choiceΔ | wall_ms mean vs control | quality/ms | kill |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| control_none | 0.000 | 0.494 | 0 | 1.00× | 0 | — |
| decode_off role_ported | 0.000 | 0.494 | 0 | ~1× | 0 | — |
| **role_ported** | **0.077** | **0.538** | **6** | **~2.6×** | mid | — |
| factor_node | 0.000 | 0.378 | 0 | ~2.5× | 0 | **apps w/o choiceΔ** |
| **direct_factors** | **0.077** | **0.538** | **6** | **~1.11×** | **highest among +acc** | — |
| lossy_pairwise | 0.000 | 0.500 | 0 | ~2.6× | 0 | **apps w/o choiceΔ** |
| role_shuffled | 0.141 | 0.530 | 49 | ~3.0× | mid | — |
| **exact_typed (0 params)** | **0.077** | **0.538** | **6** | **~1.14×** | high | — |
| oracle_diagnostic | 0.000 | 0.494 | 0 | ~1.64× | 0 | apps w/o choiceΔ |

Exact numbers: JSON `wall_ms_*`, `quality_per_ms`,
`delta_accuracy_per_extra_ms`, `wall_ms_mean_vs_control`.

### Efficiency reading

- **direct_factors** matches role_ported accuracy (+0.077) near control
  latency (~1.1×) → best quality–runtime plane among residual arms on this suite.
- **role_ported** pays ~2.6× control wall for the **same** accuracy as
  direct_factors / exact_typed — efficiency-poor here.
- **factor_node / lossy_pairwise** pay extra runtime with **zero** choice
  changes → pure cost, anti-E237 kill.
- **exact_typed** matches residual accuracy at modest cost and **0 parameters** —
  default residual until something beats it on **both** quality and ms.

## Claim verdicts

Machine-checked from metrics (`semantic_factor_claims.py`). Summary this run:
**14 validated / 4 invalidated / 1 inconclusive**.

| Claim | Verdict |
| --- | --- |
| C1–C5 safety / lossless / decode-off | **validated** |
| C6 role_ported causal | **validated** (choiceΔ>0, Δacc>0) |
| C7 factor_node causal | **invalidated** (apps without choiceΔ) |
| C8 direct_factors causal | **validated** |
| C9 role identity load-bearing | **invalidated** (shuffle ≥ role_ported) |
| C10 higher-order identity | **inconclusive** |
| C11 exact typed competitive | **validated** |
| C12 apps-without-choice kill | **validated** |
| C13–C14 math (S stochastic, soft-token collision) | **validated** |
| C15 no RL/spectral | **validated** |
| C16 promotion bar | **invalidated** (fixture) |
| **C17 runtime tracked** | **validated** |
| **C18 efficiency gate** | **validated** (slow arms without quality gain flagged) |
| **C19 exact_typed efficiency** | **invalidated** (direct_factors matches acc at lower ms) |

## Unimplemented

RL, spectral training, production topology authority, graph pruning, recurrent
semantic inference, faithful SHIFT soft tokens, Search-R1 policy training.

## Commands

```bash
python -m pytest -q \
  tests/test_harnesses/experiments/test_semantic_factor_config.py \
  tests/test_harnesses/experiments/test_semantic_factor_metrics.py \
  tests/test_harnesses/experiments/test_semantic_factor_claims.py \
  tests/test_harnesses/experiments/test_anti_e237_semantic_factor_frontier.py
python -m scripts.run_semantic_factor_frontier --seeds 0 1 2
```
