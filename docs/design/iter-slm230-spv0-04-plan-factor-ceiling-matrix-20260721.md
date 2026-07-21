# SLM-230 (SPV0-04): plan-factor oracle-substitution ceiling matrix (slm230-spv0-04-plan-factor-ceiling-matrix-20260721)

**Matrix set:** `slm230_plan_factor_ceiling_matrix`
**Version:** `spv0-04-v1`
**Status:** fixture
**Claim class:** wiring
**Corpus:** SLM-144 fixture, n=19, seed=0
**Cardinality fields populated by extractor:** False
**Gate hash:** `4e1111bab2b085a6...`
**Disposition:** ceiling_confirmed_joint_requirement — roles-only, topology-only, and bindings-only arms each stayed at or below the no-plan baseline seed-valid rate (0.00), while roles+topology raised seed-valid rate to 1.00 and adding bindings raised mean placeholder-attachment ratio from 0.00 to 1.00 (full-oracle arm: 1.00 valid rate, 1.00 attachment). This supplies the factor-wise downstream-ceiling evidence SLM-145 and SLM-160 found missing: roles and topology are jointly (not individually) sufficient to reach the structural ceiling, and bindings is jointly sufficient to reach the content ceiling on top of that -- still wiring/fixture evidence, not a learned-head promotion.

## Hypothesis

Oracle-substituting the `roles` and `topology` SemanticPlanV1 factors together (holding bindings/archetype at the empty no-plan baseline) raises PlanSeedBuilder/OpenUISemanticPlanCompiler seed validity from 0% (no-plan baseline) to a high rate on the SLM-144 fixture corpus, and additionally oracle-substituting `bindings` raises the measured content-placeholder attachment ratio from 0.0 to a high value -- supplying the factor-wise downstream-ceiling evidence SLM-145's authorization gate and the SLM-160 program disposition both found missing. Individually oracle-substituting only `roles`, only `topology`, or only `bindings` (holding the other factors at baseline) does not produce a valid seed / does not change the compiled output on any record, showing the current PlanSeedBuilder mechanism needs these factors jointly rather than accepting them as independently sufficient.

## Falsifier

Any single-factor arm (roles-only, topology-only, or bindings-only) produces a valid, non-trivial seed on a meaningful fraction of records; or the combined roles+topology arm fails to raise seed validity above the no-plan baseline; or adding the bindings factor on top of roles+topology fails to raise the placeholder-attachment ratio above the roles+topology-only arm.

## Honest caveats

- Fixture/wiring evidence only: the deterministic SLM-144 fixture corpus (`build_fixture_plan_corpus`), not a real or held-out completion corpus. No checkpoint, learned predictor, GPU run, or ship-gate claim is made or implied.
- Every oracle arm here (all but `C0_no_plan`) injects the true gold-extracted plan factor(s) directly -- this measures an upper-bound ceiling given perfect factor knowledge, not the accuracy of any predictor (none exists). `honesty_mode=oracle_diagnostic` is used throughout; none of these arms are promotable per the SemanticPlanV1 contract.
- RoleSlot.min_cardinality / max_cardinality are never populated by OpenUISemanticPlanExtractor (verified directly against the fixture corpus by this harness's `cardinality_populated` field) -- SLM-145's documented cardinality-extraction gap is confirmed still open, so no cardinality-specific arm distinct from `roles` can be measured yet.
- The `mean_seed_to_gold_ratio` token-overlap metric (reused from the sibling SLM-146/147/148 harnesses) is measured here to be dominated by PlanSeedBuilder's own statement-naming/ordering convention (`node_N`, root-last) diverging from the fixture gold renderer's convention (`nN`, root-first), independent of binding content -- it does not detect the bindings factor's marginal contribution. It is reported for continuity with those harnesses but `mean_placeholder_attachment_ratio` is the primary bindings-ceiling signal in this report.
- `mean_component_coverage` saturates at 1.0 once `roles`+`topology` are supplied, because oracle substitution injects the exact true role/component set by construction; this is expected for an oracle ceiling and is not evidence a learned predictor would reach the same coverage.

## Per-arm results

| arm | factors | n | seed valid rate | mean component coverage | mean seed-to-gold ratio | mean placeholder attachment | promotable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C0_no_plan | (none) | 19 | 0.000 | — | — | — | True |
| C1_roles_only | roles | 19 | 0.000 | — | — | — | False |
| C2_topology_only | topology | 19 | 0.000 | — | — | — | False |
| C3_bindings_only | bindings | 19 | 0.000 | — | — | — | False |
| C4_roles_topology | roles, topology | 19 | 1.000 | 1.000 | 0.215 | 0.000 | False |
| C5_roles_topology_bindings | roles, topology, bindings | 19 | 1.000 | 1.000 | 0.215 | 1.000 | False |
| C6_full_gold_oracle | archetype, roles, topology, bindings | 19 | 1.000 | 1.000 | 0.215 | 1.000 | False |

## Arm notes

- **C1_roles_only**: no valid seed on any record; reasons: expected exactly one root role, got 2, expected exactly one root role, got 3, expected exactly one root role, got 4
- **C2_topology_only**: no valid seed on any record; reasons: expected exactly one root role, got 0
- **C3_bindings_only**: no valid seed on any record; reasons: baseline: no actionable plan

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `PlanOracleSubstitutor`, `PlanSeedBuilder`, or `OpenUISemanticPlanCompiler`, does not train or evaluate a learned head, and does not flip the SLM-160 `gold_oracle_factor_heads` disposition (`retain_diagnostic`, default off). It supplies the factor-wise ceiling evidence input SLM-145's authorization gate and SLM-160's program disposition both named as the missing next step, for a human maintainer to weigh when deciding whether to reopen SLM-145.

## Reproducibility

```bash
python -m scripts.run_slm230_plan_factor_ceiling_matrix --mode plan-only
python -m scripts.run_slm230_plan_factor_ceiling_matrix --mode fixture
```
