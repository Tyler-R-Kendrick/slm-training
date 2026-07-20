# SPV4-02: Causal architecture disposition report (SLM-160)

**Schema:** `SPVDispositionV1`

**Matrix set:** `slm160_spv_disposition`

**Version:** `spv4-02-v1`

**Status:** fixture

**Claim class:** wiring / disposition audit only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

**Canonical artifact pair:**

- Machine-readable disposition: [`iter-slm160-spv-disposition-20260720.json`](iter-slm160-spv-disposition-20260720.json)
- Generated narrative: [`iter-slm160-spv-disposition-20260720.md`](iter-slm160-spv-disposition-20260720.md)

This document is the stable canonical report required by SLM-160; the
`iter-*.json` / `iter-*.md` files are the timestamped, machine-readable and
human-readable artifacts produced by the disposition harness and are the source
of truth for the structured evidence.

## Executive finding

SPV4-02 audit covers semantic planning, valid-state init/search, scoring/supervision, generation factorization, and portability. All evidence up through SLM-159 is wiring/fixture, blocked, or measured-not-promotable. No mechanism satisfies the criteria for adopt_primary or adopt_optional. The GraphQL pack replication is a retained diagnostic; all second-pack portability is blocked pending real pack implementations.

## Evidence chronology

The audit aggregated the following committed ``docs/design`` artifacts:

- `docs/design/iter-slm146-semantic-plan-compiler-20260720.json`
- `docs/design/iter-slm144-plan-predictor-20260720.json`
- `docs/design/iter-slm145-plan-predictor-factors-20260720.json`
- `docs/design/iter-slm147-x22-retrieval-20260720.json`
- `docs/design/iter-slm148-x22-conflict-campaign-20260720.json`
- `docs/design/iter-slm154-legal-action-scorer-20260720.json`
- `docs/design/iter-slm155-factorization-comparison-20260720.json`
- `docs/design/iter-spv2-02-global-semantic-critic-20260720.json`
- `docs/design/iter-spv0-03-semantic-regret-20260719.json`
- `docs/design/iter-slm120-corruption-curriculum-20260719.json`
- `docs/design/iter-spv2-03-legal-set-distillation-20260720.json`
- `docs/design/iter-spv2-04-dense-teacher-mixture-20260720.json`
- `docs/design/iter-spv2-05-semantic-repair-20260720.json`
- `docs/design/iter-slm156-plan-refinement-20260720.json`
- `docs/design/iter-slm158-mixer-comparison-20260720.json`
- `docs/design/iter-slm157-flow-consistency-20260720.json`
- `docs/design/iter-slm159-cross-dsl-replication-20260720.json`
- `docs/design/iter-e575-prompt-semantic-plan-soft-20260720.json`
- `docs/design/iter-e576-prompt-plan-binding-soft-20260720.json`
- `docs/design/iter-e579-verified-plan-root-20260720.json`

## Mechanism disposition table

| Mechanism | Issues | Disposition | Default state | Rationale |
| --- | --- | --- | --- | --- |
| semantic_plan_v1_ir | SLM-43, SLM-71, SLM-146 | retain_diagnostic | n/a | Contract/spec artifact; no predictor claim. GraphQL pack demonstrates factor extraction, but predicted-plan generalization is unproven. |
| gold_oracle_factor_heads | SLM-144, SLM-145 | retain_diagnostic | off | Gold oracle arms provide diagnostic ceilings, but SLM-145 closed without factor-wise gold-substitution evidence for topology, cardinality, or bindings; learned heads are not justified. |
| plan_seed_builder_soft_restrictions | SLM-146 | retain_diagnostic | off | Fixture-only synthetic corpus; no production decoder. Unsafe predicted-hard arm is explicitly non-promotable. |
| x22_seed_retrieval_conflict_repair | SLM-147, SLM-148 | retain_diagnostic | off | Fixture-only evidence; no live X22 model trained or decoded. Oracle and gold-plan arms are diagnostic ceilings only. |
| ar_legal_action_scorer | SLM-154, SLM-155 | retain_diagnostic | off | Wiring-only fixture; no live production decode loop and no ship readiness claim. SLM-154 explicitly labels itself fixture wiring. |
| global_semantic_critic | SLM-150, SPV2-02 | retain_diagnostic | off | Synthetic fixture corpus only; the SPV2-01 hard-valid contrast corpus is absent from the repo. |
| hard_valid_contrasts | SPV0-03, SLM-120 | retain_diagnostic | off | SPV0-03 is a fixture regret diagnostic; SLM-120 is a frontier curriculum plan. No production contrast corpus has been built. |
| dense_legal_set_distillation | SPV2-03, SPV2-04 | retain_diagnostic | off | Both SPV2-03 and SPV2-04 are fixture wiring only; no external teacher model, solver replay, or checkpoint training was performed. |
| semantic_repair | SPV2-05 | retain_diagnostic | off | Wiring-only fixture baseline; real verifier-backed counterfactual action values require SLM-131/VSS finite replay. |
| plan_refinement_slm156 | SLM-156 | retain_diagnostic | off | Fixture-only synthetic plan-state recovery; no downstream completion or ship-gate evidence. |
| mixer_slm158 | SLM-158 | retain_diagnostic | off | Fixture-only synthetic token-pattern classifier; no OpenUI completion or ship-gate evidence. |
| flow_consistency_slm157 | SLM-157 | blocked | blocked | Blocked: upstream dependencies SLM-99/SLM-148 are not done and no implementation exists. The evidence document is also absent. [Evidence missing at audit time: docs/design/iter-slm157-flow-consistency-20260720.json] |
| multi_pack_graphql | SLM-159 | retain_diagnostic | off | GraphQL replication fixture succeeds, but it is wiring-only with no predictor claim and no ship-gate evidence. |
| multi_pack_second_pack | SLM-159, SLM-44, SLM-45 | blocked | blocked | No SLM-44 or SLM-45 pack is registered; only design documents exist. A syntax-only toy pack cannot satisfy the readiness rubric. |
| prompt_plan_soft_scoring_e575_e576_e579 | E575, E576, E579 | retain_diagnostic | off | E575/E576/E579 report local/structural gains but are explicitly not promotable: binding-aware meaning-v2 and AgentV remain zero. |

## Cross-pack summary

SPV4-02 audit covers semantic planning, valid-state init/search, scoring/supervision, generation factorization, and portability. All evidence up through SLM-159 is wiring/fixture, blocked, or measured-not-promotable. No mechanism satisfies the criteria for adopt_primary or adopt_optional. The GraphQL pack replication is a retained diagnostic; all second-pack portability is blocked pending real pack implementations.

## Canonical architecture recommendation

Canonical architecture remains the existing honest-slot-contract TwoTower decoder with all plan-aware mechanisms retained as default-off diagnostics. Do not promote or sync checkpoints from E575/E576/E579. Unblock flow/consistency (SLM-157) and second-pack portability only after their upstream dependencies are implemented and measured. The next high-leverage step is a factor-wise oracle-substitution matrix for topology/cardinality/bindings and a versioned hard-valid semantic-contrast corpus.

## Rejected or blocked mechanisms

- `flow_consistency_slm157`
- `multi_pack_second_pack`

## Reproducibility commands

```bash
# Plan-only manifest (no evidence reads)
python -m scripts.run_slm160_spv_disposition --mode plan-only

# Fixture audit that reads docs/design evidence and writes the report
python -m scripts.run_slm160_spv_disposition --mode fixture
```

## Limitations

- This report is a docs/spec audit, not a training or evaluation run.
- Dispositions are conditioned on the evidence available up to the cutoff commit; new measured results can change them.
- Any evidence file marked missing downgraded the associated mechanism to ``blocked`` or ``inconclusive``.
- No ship-gate claim is made; no mechanism is promoted to ``adopt_primary``.
