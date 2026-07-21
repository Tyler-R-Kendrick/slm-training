# SLM-189 (FFE2-01): bridge planner fixture (slm189-bridge-planner-20260721)

Matrix set: `slm189_bridge_planner`

Version: `ffe2-01-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU, no model, no checkpoint, and no ship-gate claim is made.

## Hypothesis

A deterministic bridge planner can produce replay-valid canonical edit sequences from structural sketch seeds to fixture OpenUI targets within a small edit budget, and independent-edit permutations that respect the dependency DAG preserve reachability and transition validity.

## Falsifier

The canonical greedy arm fails to reach a supported fixture target, or a dependency-respecting permutation of the greedy edits fails to replay to the target, or the exact-shortest arm disagrees with the greedy arm on tiny cases where it should be feasible, or any arm reports certificate failure for a reachable target.

## Arms

| arm_name | n_cases | n_reached | n_unknown_budget | mean_path_length | p95_path_length | source_bias_index | excess_cost_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| canonical_greedy | 6 | 6 | 0 | 3.00 | 4.00 | 0.0000 | 1.0000 |

## Cases

Total cases: 6
Reached: 6
Source policies: minimal

| case_id | source_seed_id | target_id | arm | status | path_length | replay_ok |
| --- | --- | --- | --- | --- | --- | --- |
| minimal__hero_card__canonical_greedy | minimal | hero_card | canonical_greedy | reached | 3 | True |
| minimal__simple_stack__canonical_greedy | minimal | simple_stack | canonical_greedy | reached | 1 | True |
| minimal__card_with_button__canonical_greedy | minimal | card_with_button | canonical_greedy | reached | 4 | True |
| minimal__button_row__canonical_greedy | minimal | button_row | canonical_greedy | reached | 4 | True |
| minimal__nested_stack__canonical_greedy | minimal | nested_stack | canonical_greedy | reached | 2 | True |
| minimal__image_card__canonical_greedy | minimal | image_card | canonical_greedy | reached | 4 | True |

## Disposition

**inconclusive**

exact_shortest mismatched canonical_greedy on 6/6 tiny cases; the exact BFS arm is not fully exercised in this fixture.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. The bridge planner arms, dependency DAG, transition certificates, and source-policy variation are exercised over deterministic synthetic targets, but no real model or decode path was run. The mechanism remains ``retain_diagnostic`` / ``blocked_pending_real_model`` until trained-model bridge telemetry and AgentV evaluation are available.

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.
- exact_budget is intentionally small (8) so the harness stays CPU-only; production bridge search needs a larger solver budget.
- The production corpus is not exercised; targets are hand-written or deterministically generated small OpenUI programs.
- solver_guided, contract_first, and source_adaptive arms are documented but not implemented in this wiring fixture; they return UNKNOWN_BUDGET.
- Randomization in random_shortest is over topological orders of the dependency DAG only, not over the full edit search space.

## Reproducibility

```bash
python -m scripts.run_bridge_planner_audit --mode plan-only
python -m scripts.run_bridge_planner_audit --mode fixture
```
