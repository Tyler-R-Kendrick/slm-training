# SLM-159 (SPV4-01): Cross-DSL semantic-plan replication fixture (slm159_fixture)

Matrix set: `slm159_cross_dsl_replication`

Version: `spv4-01-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no production TwoTower wiring was touched, and no ship-gate claim is made.

## Hypothesis

Pack-neutral SemanticPlanV1 extraction, seed construction, and oracle-backed validation transfer from OpenUI to GraphQL and a structurally different second pack.

## Falsifier

Plan factors cannot be defined from GraphQL's schema/selection semantics, the seed builder cannot reproduce schema-valid queries, or the second-pack candidates lack the grammar/parser/oracle/data contract required for a non-toy replication.

## Arms

| Arm | Pack | Family | Promotable | Blocked | Description |
| --- | --- | --- | --- | --- | --- |
| G1_graphql | graphql | graphql | True | False | Replicate SemanticPlanV1 extraction and seed building on the SLM-43 GraphQL pack, using the schema as the symbol table. |
| S1_second_pack | design-patterns-or-nomenclature | second_pack | False | True | Preregistered second-pack candidate from SLM-44 (design-patterns DSL) or SLM-45 (expert-nomenclature/ontology pack).  The issue authorizes recording a concrete readiness blocker if neither pack/oracle contract is available. |

## Pack readiness

| Pack | Available | Parser | Oracle | Generator | Canonicalizer | Placeholder | Pass | Blocker |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| graphql | True | True | True | True | True | True | True | - |
| design-patterns | False | False | False | False | False | False | False | pack 'design-patterns' is not registered in the DSL pack registry |
| nomenclature | False | False | False | False | False | False | False | pack 'nomenclature' is not registered in the DSL pack registry |
| ontology | False | False | False | False | False | False | False | pack 'ontology' is not registered in the DSL pack registry |
| expert-nomenclature | False | False | False | False | False | False | False | pack 'expert-nomenclature' is not registered in the DSL pack registry |

## Results

| Arm | Seed | Records | Extraction | Seed validity | Round-trip | Latency ms |
| --- | --- | --- | --- | --- | --- | --- |
| G1_graphql | 0 | 4 | 1.00 | 1.00 | 1.00 | 198.170 |
| S1_second_pack | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.000 |

## Verdict

This is a fixture wiring run. GraphQL plan extraction and seed building are exercised through the pack's own oracle; the second-pack replication is intentionally blocked until SLM-44 or SLM-45 provides a real grammar/parser/oracle/data contract. Real claims require pack-native semantic metrics, causal plan-oracle substitution, learned plan recovery, and independent ship-gate evaluation.
