# SLM-511 (PGS-H02): Goal-support domain-adequacy fixture campaign evidence

Design / authority / threat model:
[`proof-carrying-goal-support.md`](proof-carrying-goal-support.md) (PGS-I01 / SLM-512).

**Harness:** SLM-510 / PGS-H01 (preregistered `ExperimentCampaignV1`)

**Claim class:** `diagnostic` — fixture/wiring evidence only (no train, checkpoint, ship, promotion, MODEL_CARD)

**Campaign:** `pgs-h01-goal-support-domain-adequacy-fixture`

**Git commit:** `1696e52a9f42934f0254a3557b9774668f0025b9` (clean worktree)

**Manifest digest:** `3b8c082a04bab1191048dfda4587612ff3e24b01f6968fe7bf281f93c719e951`

**Primary metric (`exact_action_coverage_rate`, production-exact arm mean):** 0.977778

**Result digest:** `f4c1baac277fff157b9f384cd30a85d6156909e60677f0404408fa4e24945268`

**Deterministic rerun digest:** `f4c1baac277fff157b9f384cd30a85d6156909e60677f0404408fa4e24945268` (match)

## Pre-run admission

| Check | Value |
| --- | --- |
| worktree_clean | True |
| manifest_sha256 | `3b8c082a04bab1191048dfda4587612ff3e24b01f6968fe7bf281f93c719e951` |
| fixture_count | 15 |
| arms | structural_reference, production_exact, evaluation_oracle, certified |
| exact_action_cap | 2..10 (default 10; `cap_excludes_unique_support`=2) |
| default bounds_digest | `2b361facd10ed3b6e863d366dd62fc273cdc5de1e7d523ad51e4cd594a9d7ea5` |
| adversarial/decode tests | 98 passed (goal_support adversarial + decode_g03 + campaign harness) |
| post-prereg fixture/evaluator change | None (manifest identity unchanged) |

Schema/implementation pins: `goal_support_domain_adequacy_campaign_fixture/v1`, `goal_verifier_profile/v1`, `goal_terminal_evidence/v1`, `goal_support_result/v1`, `compiled_goal_constraint_set/v1`, `goal_support/v1`, constraint `v1`, compiler `compiler/v1`.

## Acceptance snapshot

| Check | Value |
| --- | --- |
| false_hard_prune_count | 0 |
| support_certificate_replay_failure_count | 0 |
| evaluation_oracle_called_certified_closure | False |
| certified_singleton_zero_forward (fixture) | forwards=0, replay_fail=0 |
| deterministic_rerun_digest_match | True |
| promotion | False |

Note: aggregate `falsifier_holds=True` reflects `bound_exhaustion` certified arm doing forwards without removal under tight bounds (`certified_singleton_forward_violations=1`); primary kill counters above remain zero.

## Arms (aggregate means)

| Arm | exact_action_coverage_rate | domain_inadequate_rate | coverage_unknown_rate | false_hard_prune | replay_failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| structural_support_reference | 1.0 | 0.0 | 0.0 | 0 | 0 |
| goal_support_production_exact_diagnostic | 0.977778 | 0.133333 | 0.4 | 0 | 0 |
| goal_support_evaluation_oracle_diagnostic | 0.977778 | 0.0 | 0.533333 | 0 | 0 |
| goal_support_certified_fixture | 0.974359 | 0.153846 | 0.384615 | 0 | 0 |

Additional H01 counters (production-exact arm): selection_regret_rate=0.066667, selection_unresolved_rate=0.0, structural_supported_but_goal_unsupported_rate=0.0, obstruction_core_emission_rate=0.133333, obstruction_core_replay_rate=0.133333, mean_obstruction_core_size=0.133333, verifier_calls_total=27, expanded_nodes_total=42, wall_time_ms_total≈30.2.

## Independent verification highlights

| Scenario | Evidence |
| --- | --- |
| Base certs + goal sidecars replay | `bounded_domain_inadequacy` / `candidate_addition_stale_identity`: obstruction_core_emitted=1, replay_ok=1 under production-exact |
| Exact obstruction cores + mode | `domain_inadequate_under_bounds` on inadequate fixtures; no core on unknown/partial/eval/advisory |
| Unknown/partial/eval/advisory | `coverage_unknown`; certified arm skipped for advisory + evaluation_oracle fixtures |
| Candidate cap → coverage_unknown | `cap_excludes_unique_support`: classification `coverage_unknown` |
| Certified singleton bypass | `certified_singleton_zero_forward`: certified_singleton_forwards=0, verifier_calls=1, expanded_nodes=1 |
| Evaluation-oracle never prunes | eval arm: unsupported_action_rate=0; cannot call `exact_goal_closure` (arm isolation test green) |
| Diagnostic preserves forest identity | diagnostic_domain_digest_changes=0 |

## Recipe

```bash
python -m scripts.run_goal_support_domain_adequacy --mode fixture \
  --out-dir outputs/runs/pgs_h02_goal_support_domain_adequacy \
  --docs-out docs/design/proof-carrying-goal-support-fixture-results.json \
  --claim-class diagnostic
```

Device: CPU. Honesty mode: `fixture_diagnostic`. `version_stamp.components.harness.experiments`: `v163`.

Alias (H01 CLI default name): `docs/design/iter-slm510-pgs-h01-goal-support-domain-adequacy-20260811.json` → primary JSON above.

Full machine-readable detail: `docs/design/proof-carrying-goal-support-fixture-results.json`.

## Honest disposition

Fixture/diagnostic wiring evidence for goal-support domain adequacy under complete finite word-tree fixtures. Does **not** certify model quality, semantic generalization, ship gates, promotion, global UNSAT, or production latency. Negative/limitation cases (bound exhaustion, cap exclusion, partial forest) are recorded as bounded classifications — not retuned.
