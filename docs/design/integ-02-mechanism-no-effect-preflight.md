# INTEG-02 — Theorem-backed trigger/no-effect admission (SLM-555)

## Claim

Optional fail-closed autoresearch preflight may **skip** an expensive mechanism
treatment only when a KERN-08 necessary-trigger theorem discharges **no-effect**
over the exact locked corpus with **complete** trigger-absence evidence and a
**complete** scan. Present triggers, unknown evidence, incomplete scans, and
mechanisms without a safe theorem always **run**.

## Adapter (no new scheduler)

| Surface | Role |
| --- | --- |
| `autoresearch/preflight/mechanism_no_effect.py` | Admission + discovered `CHECK` plugin |
| `admit_mechanism_treatment` / `admit_for_campaign` | Replayable `skip_no_effect` / `run` disposition |
| `admit_for_campaign(manifest, ...)` | Bind disposition to ExperimentCampaignV1 identity (HARN-10) |
| KERN-08 `formal/mechanism_trigger.py` | Sole certificate authority |

Schema: `mechanism_no_effect_admission/v1`. Skip maps to preflight `block`;
run maps to `pass`. Missing `mechanism_id` leaves the check idle.

## Disposition rules

| Evidence | Disposition |
| --- | --- |
| Complete scan ∧ all triggers `absent` ∧ safe theorem ∧ certificate emits | `skip_no_effect` |
| Any trigger `present` (successful / active treatment) | `run` |
| Any trigger `unknown` / incomplete row evidence | `run` |
| `scan_complete=False` (or omitted on candidate) | `run` — never infer from zero observed activation |
| No safe trigger theorem / unsupported mechanism | `run` |

## Acceptance

- Fixture / historical successful treatments with a present necessary trigger
  are never skipped.
- Inactive treatments with complete trigger-absence evidence skip
  deterministically (stable `content_sha256`).
- Unknown or incomplete evidence always falls back to running.

## Lineage

Disposition payloads bind `campaign_id`, `experiment_id`, `manifest_sha256`,
observation digest, and certificate digest. Controls and campaign identity are
not rewritten — admission only decides spend vs skip.

## Tests

`tests/test_autoresearch/test_mechanism_no_effect_preflight.py`.
