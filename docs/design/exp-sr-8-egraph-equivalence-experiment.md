# EXP-SR-8: proof-scoped e-graph/equivalence-region experiment

Campaign manifest: `78b26b538843a36d276167eaad698c25f901d57fc3ade023ae69e42ed7455c86` (claim_class=`fixture`).
Fixture: 6 known-equivalence groups, 4 known-distinct pairs, 20 raw input forms.

## Control arm (canonical-hash dedup alone)

- Unique canonical forms: 7
- Wall time: 1.110 ms

## Candidate arm (e-graph saturation + extraction)

- Unique extracted forms: 7
- e-nodes: 14, e-classes: 9
- Saturation rounds: 2
- Rewrite applications: 10
- Wall time: 6.150 ms

## Comparison

- Unique-form sets match: True
- Pareto frontiers match: True
- Wall-time overhead (candidate - control): 5.040 ms

## Verification (EXP-SR-8 primary endpoint)

- Rewrite applications independently re-verified: 10
- Passed: 10
- `egraph_certified_equivalence_rate`: 1.0

## Gates

- `exp-sr-8_promotion` (ge 0.0, value=1.0): FIRED
- `exp-sr-8_kill` (lt 1.0, value=1.0): clear

**Adoption decision:** `not_adopted_stays_experimental`

e-graph equivalence regions matched canonical-hash dedup's unique-form set and Pareto frontier exactly on this fixture (unique_forms_match=True, pareto_frontier_matches=True); every certified rewrite union re-verified numerically. For SRP-005's current REWRITE_RULES (each rule is complexity-non-increasing and the rule set is confluent), e-graph saturation is mathematically guaranteed to converge to the same fixed point plain canonicalize_expr already computes, so no quality/cost frontier improvement over simpler canonicalization is possible with this rule set. Stays an isolated, default-off experiment module regardless: no import from dsl/symbolic_regression_pack.py or any production canonicalization path.

Full detail: `docs/design/exp-sr-8-egraph-equivalence-experiment.json`.
