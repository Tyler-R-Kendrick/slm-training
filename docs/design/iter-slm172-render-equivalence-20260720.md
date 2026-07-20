# SLM-172 (SDE2-05): render-equivalence fixture (slm172-render-equivalence-20260720)

Matrix set: `slm172_render_equivalence`

Version: `sde2-05-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

Canonical AST signature, normalized render-tree overlap, and optional visual-diff surrogate agree on semantic equivalence and reject structural corruptions.

## Falsifier

A structural corruption (topology, binding, component substitution) is marked equivalent, or a canonical exact match is rejected.

## Render-equivalence arms

| arm_id | arm_name | seed |
| --- | --- | --- |
| canonical_exact__s0 | canonical_exact | 0 |
| alpha_renamed__s0 | alpha_renamed | 0 |
| style_only_change__s0 | style_only_change | 0 |
| topology_corruption__s0 | topology_corruption | 0 |
| binding_corruption__s0 | binding_corruption | 0 |
| component_substitution__s0 | component_substitution | 0 |
| metric_gaming_minimal_valid__s0 | metric_gaming_minimal_valid | 0 |

## Results

| arm_id | arm_name | seed | canonical_exact | binding_graph_equal | component_type | role | topology | cardinality | binding_graph | interaction_dep | render_tree_dist | tier2_status | equivalent | wall_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_exact__s0 | canonical_exact | 0 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | not_available | 1 | 0.003 |
| alpha_renamed__s0 | alpha_renamed | 0 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | not_available | 1 | 0.003 |
| style_only_change__s0 | style_only_change | 0 | 1 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | not_available | 1 | 0.003 |
| topology_corruption__s0 | topology_corruption | 0 | 0 | 1 | 0.857 | 1.000 | 0.400 | 0.750 | 1.000 | 1.000 | 0.500 | not_available | 0 | 0.023 |
| binding_corruption__s0 | binding_corruption | 0 | 0 | 1 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | not_available | 0 | 0.022 |
| component_substitution__s0 | component_substitution | 0 | 0 | 1 | 0.667 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.667 | not_available | 0 | 0.002 |
| metric_gaming_minimal_valid__s0 | metric_gaming_minimal_valid | 0 | 0 | 0 | 0.400 | 0.000 | 0.000 | 0.667 | 0.000 | 1.000 | 0.333 | not_available | 0 | 0.024 |

## Per-arm means

| arm_name | equivalent_rate | canonical_exact | binding_graph_equal | component_type | role | topology | cardinality | binding_graph | interaction_dep | render_tree_dist |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_exact | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| alpha_renamed | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| style_only_change | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| topology_corruption | 0.000 | 0.000 | 1.000 | 0.857 | 1.000 | 0.400 | 0.750 | 1.000 | 1.000 | 0.500 |
| binding_corruption | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| component_substitution | 0.000 | 0.000 | 1.000 | 0.667 | 0.500 | 0.500 | 1.000 | 1.000 | 0.000 | 0.667 |
| metric_gaming_minimal_valid | 0.000 | 0.000 | 0.000 | 0.400 | 0.000 | 0.000 | 0.667 | 0.000 | 1.000 | 0.333 |

## Disposition

**calibrated**

Canonical exact, alpha-renamed, and style-only pairs are equivalent, while all structural corruptions and metric-gaming traps are rejected.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The render-equivalence surrogates are exercised over deterministic synthetic and contrast pairs, but no real model was trained or evaluated. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until a trained model and AgentV evaluation are available.

## Honest caveats

- Tier-2 visual diff is capability-gated; in most CI/fixture environments it   reports ``not_available``.
- Semantic-contrast pairs come from a small deterministic builder corpus, not   a trained model or real user distribution.
- Component substitution and metric-gaming cases are hand-selected traps, not   a representative sample.
- No ship-gate claim is made; this is wiring evidence only.

## Reproducibility

```bash
python -m scripts.run_slm172_render_equivalence_fixture --mode plan-only
python -m scripts.run_slm172_render_equivalence_fixture --mode fixture
```
