# INTEG-03 — Campaign formal/empirical evidence linkage (SLM-556)

## Claim

Four-axis formal analysis (EVID-03), FormalAuthorityV2 (EVID-06), and revmath
task/report digests (HARN-02/HARN-10) are first-class **optional** linked
evidence on existing `CampaignResultV1` / dispositions. Formal and empirical
statuses persist independently. A proof cannot set ship or promotion success
by itself. Historical campaign locks and readers stay byte-compatible.

## Adapter (extend owners; no parallel campaign DB)

| Surface | Role |
| --- | --- |
| `autoresearch/campaign_formal_evidence.py` | Split schema, link builder, migration, decision-support report |
| `autoresearch/experiment_campaign.py` | Optional `CampaignResultV1.formal_empirical`; separation check in `validate_result_claim` |
| `harnesses/reasoning/revmath/profile_binding.py` | HARN-10 profile runs attach content-bound refs + write decision-support artifacts |

Schema: `campaign_formal_empirical_split/v1`. Refs are content-SHA bound
(`formal_authority_v2`, `formal_preflight`, `four_axis_ledger`,
`revmath_report`, `revmath_result`, `bound_ast`, optional
`runtime_refinement_trace` slot for KERN-11, `empirical_remainder_claim`).

## Separation rules

| Formal | Empirical | Ship / promotion |
| --- | --- | --- |
| proved / refuted | failure / inconclusive | **blocked** — proof does not stamp `ship_gates_passed` |
| unknown / weak / unchecked / absent | success | **allowed** — empirical path alone may succeed |
| any | success | empirical status must be `success` for ship stamp |
| absent (historical) | n/a | unchanged readers; `formal_empirical=None` |

`ExperimentCampaignV1` lock digests are untouched — the split lives only on
results/evidence, never in the preregistered plan bytes.

## Query surface

`decision_support_report(split)` returns which assumption / computability /
resource-bound / refinement axis statuses and which digests support the
decision. Profile runs also persist
`campaign_decision_support_reports` under `CampaignStore`.

## Migration

`migrate_campaign_result_payload` leaves missing `formal_empirical` unset and
seals digests on mixed-version bodies that omit `content_digest` without
inventing axis claims.

## Tests

`tests/test_autoresearch/test_campaign_formal_evidence.py`.
