# VCE-005 (SLM-458): intervention control arms

Extends [`plan_oracle_substitutor`](repository-ownership-map.md) (VCE-004/SLM-447,
`src/slm_training/data/semantic_plan/oracle.py`) with the remaining control arms an
oracle-localization result needs so it cannot be mistaken for a generic
extra-compute, leakage, or intervention-pipeline effect. No new record schema, no
second substitution authority: every arm below still produces a
`PlanInterventionRecordV1` via the existing `apply_plan_intervention`.

## What was already there (VCE-004) vs. what this adds

| Arm | Status before VCE-005 | This change |
| --- | --- | --- |
| Baseline / no-op | Incidental (`plan_source="none"` fallthrough) | `build_baseline_intervention` — explicit, first-class, same telemetry bookkeeping as every other arm |
| One-factor / all-oracle | Implemented | Unchanged |
| Shuffled (different compatible record) | Missing | `select_shuffled_oracle` — content-only compatibility key, reuses `PlanOracleSubstitutor.apply` unchanged once a candidate is picked |
| Destructive (wrong value, matched work) | Missing | `PlanSource` extended with `"destructive"`; `PlanOracleSubstitutor.apply` permutes baseline's own existing per-factor content |
| Random / advisory perturbation | Missing | `PlanSource` extended with `"random"`; same permutation machinery, seeded |

## Design: permutation, not fabrication

Both new synthetic sources reuse one idea: a destructive/random factor value is a
**permutation of the baseline's own existing content**, never new content and never
another record's content.

- `archetype`: swapped to a different, fixed (`destructive`) or seed-chosen
  (`random`) id from a small alternates list.
- `roles` / `bindings`: the existing `component_family` / `candidate_symbols`
  values are reordered across the existing role slots / bindings (same element
  count).
- `topology`: the existing edges' `parent_role_id` values are reordered across
  the existing edges (same edge count).

This directly satisfies "destructive/wrong factor mutation with **matched
serialization/work**": the mutated plan has exactly as many role slots, edges, and
bindings as the baseline, so its downstream compute/serialization cost is
comparable to a real one-factor substitution of the same factor — not a
distinguishable "did more work" tell. Verified in
`test_destructive_arm_changes_exactly_the_declared_factor_with_matched_shape`
(asserts `len(role_slots)`/`len(topology.parent_relation_candidates)`/`len(bindings)`
unchanged) and `test_random_arm_matches_destructive_arms_shape_and_banner`.

When a factor has too little structure to permute (e.g. one role slot, one
binding), the permutation returns `None` and `apply()` leaves that factor
unchanged — the existing "unknown factors preserve baseline behavior" convention,
not a fabricated change. Covered by
`test_synthetic_arms_fall_back_to_no_op_when_nothing_to_permute`.

Neither `"destructive"` nor `"random"` is contamination-bannered
(`contamination_banner()`'s gate, `{"gold", "oracle_override"}`, is unchanged) --
they never read real oracle content, so they aren't oracle leakage and can safely
enter a manifest via `filter_manifest_safe`, unlike a `plan_source="gold"` shuffled
pairing (which correctly *is* bannered by the same, unchanged gate — see
`test_shuffled_oracle_intervention_is_gated_by_existing_contamination_rules`).

## Shuffled: content-only compatibility, no identity leakage

`select_shuffled_oracle(baseline, candidates, rng_seed=...)` picks a different plan
from `candidates` using `_compatibility_key` -- `(archetype.id, len(role_slots),
len(symbols), len(bindings))` -- and nothing else. It never reads
`identity.source_program_fingerprint` or `identity.prompt_context_hash`. Once
picked, the shuffled plan is passed to the unmodified `PlanOracleSubstitutor.apply`,
so its contamination status is decided by the caller's declared `plan_source`
exactly as for a real oracle pairing.

`test_select_shuffled_oracle_never_uses_identity_fields_to_choose` directly proves
this: two plans with identical content but different fingerprints are still a
valid, distinct shuffle pair, so the selection logic provably cannot be using
those fields to choose or exclude a candidate.

## Evidence

`pytest tests/test_data/test_semantic_plan_extraction/test_oracle.py` -- 32/32
passing (the 16 pre-existing VCE-004 tests, unchanged, plus 16 new). Also verified
no regression in the one other `PlanOracleSubstitutor` consumer,
`harnesses/experiments/slm148_x22_conflict_campaign.py`
(`tests/test_harnesses/experiments/test_slm148_x22_conflict_campaign.py`, 14/14
passing) -- its calls are positional `apply(baseline, oracle)`, unaffected by
`oracle` becoming optional / `rng_seed` becoming keyword-only.

## Honest scope

- "Random/advisory perturbation" is genuinely random only when a factor has
  enough elements for its permutation space to have more than one non-identity
  ordering (e.g. 2-element lists have exactly one possible swap, so every seed
  produces the same result there) -- `test_random_arm_is_seed_sensitive_across_enough_seeds`
  checks sensitivity across ten seeds on `archetype` (six alternates) rather than
  claiming seed-sensitivity is guaranteed for every factor/plan shape.
- This issue does not touch `oracle.py`'s existing one-factor/all-oracle/no-op
  paths beyond making `oracle` optional and adding `rng_seed` -- both additive,
  backward-compatible signature changes.
- No new persisted campaign/corpus of intervention records was built here; this
  is harness code plus unit tests, consistent with VCE-004's own scope (the
  campaign-scale execution is SIE-002/SLM-476, still blocked on other work).
