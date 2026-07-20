# SLM-143 / SPV0-03: Bounded completion enumeration and semantic regret decomposition

**Claim class:** wiring / fixture only  
**Run date:** 2026-07-19  
**Machine-readable result:** [`iter-spv0-03-semantic-regret-20260719.json`](iter-spv0-03-semantic-regret-20260719.json)

This iteration wires the SLM-143 semantic-regret decomposition harness. No
OpenUI checkpoint was evaluated, no GPU was used, and no ship-gate claim is
made.

## What landed

- `src/slm_training/harnesses/experiments/semantic_regret_matrix.py`
  - Frozen dataclasses: `BoundedCompletionState`, `CompletionSnapshot`,
    `RegretMetrics`, `RegretReport`, `SemanticRegretMatrixReport`.
  - Deterministic bounded completion enumerator
    (`enumerate_bounded_completions`).
  - Trace-based regret decomposition (`compute_regret_from_trace`):
    representation regret, candidate coverage, acceptable-action rank,
    local regret, pruning regret, global-rank regret, and plan-regret
    placeholder.
  - `plan_regret_delta` for per-factor delta reporting.
  - Adapter placeholders for compiler-choice, x22, and selector candidate-set
    regret decomposition.
- `scripts/run_semantic_regret_fixture.py`
  - Builds a deterministic toy graph with known exact regrets.
  - Computes greedy, oracle, and representation-regret arms.
  - Uses `PlanOracleSubstitutor` with two `SemanticPlanV1` instances to
    demonstrate factor-wise plan substitution for the `archetype` factor.
- Tests under `tests/test_harnesses/experiments/test_semantic_regret_matrix.py`
  and `tests/test_scripts/test_run_semantic_regret_fixture.py`.
- Registry entries: `harness.experiments` bumped and a new
  `harness.experiments.semantic_regret` v1 component.

## Fixture results

The toy graph contains:

- an accepted reachable branch (`accept_good` = 1.0, `accept_ok` = 0.6),
- a pruned high-value branch (`prune_high` = 1.5, `prune_cause="budget"`),
- a globally best accepted completion (`target` = 2.0) reached via
  `continue` -> `mid`,
- a disconnected unreachable target (`oracle_best` = 2.0) used to exercise
  representation regret.

Key decompositions are in the linked JSON. The fixture demonstrates that:

- choosing any accepted action yields zero local regret,
- a pruned action does not inflate scoring regret,
- an unreachable oracle-best target produces `UNKNOWN` representation regret,
- substituting the oracle `archetype` factor changes the plan-regret delta
  terms.

## Exact command

```bash
python -m scripts.run_semantic_regret_fixture
```

## Honest verdict

**`no_safe_direction` / wiring-only.** The harness compiles, the regret
terms are defined and computed on a toy graph, and the plan-factor
substitution wiring is exercised. The fixture is too small and too artificial
to tell whether the decomposition will generalize to real OpenUI completion
enumeration. A production claim would require:

- A real grammar/constrained-decode completion enumerator,
- A trained OpenUI model producing scored candidate sets,
- Oracle plans derived independently from gold programs,
- Held-out honest ship-gate suites, and
- An explicit audit that no hidden gold channel leaks into the regret terms.

Until then this is wiring and a reusable diagnostic harness, not a ship
result.
