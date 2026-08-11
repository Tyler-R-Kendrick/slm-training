# INTEG-01 — Canonical proof-trace projection (SLM-528)

## Claim

Existing runtime evidence (`ReplayBundleV1`, `DecodeStats` /
`DecodeStatsRecordV1`, `MechanismActivationV1`, `VerifierWitnessV1`) projects
deterministically into one canonical proof-trace vocabulary that reuses
KERN-06 `Event` / named machine-cost models for observed work. The projection
**does not** introduce a second telemetry recorder and **does not** claim
abstract refinement — KERN-11 (SLM-539) proves refinement separately.

## Canonical fields

| Field | Typical owner |
| --- | --- |
| `state_input_identity` | `ReplayBundleV1.identity` / `DecodeIdentityV1` |
| `legal_domain` | bundle legal-domain size/status/digest |
| `ranker_model_invocation` | `DecodeStats.forwards_count` |
| `chosen_action` | `decoder_choice` / solver `decision` |
| `verifier_solver_certificate` | solver certificate events / `VerifierWitnessV1` |
| `cache_mechanism` | `artifact_state` / `MechanismActivationV1` |
| `state_transition` | `solver_state` / `backtrack` |
| `observed_work` | KERN-06 `project_decode_stats` unit-work projection |

Unobserved fields stay listed in `unobserved_fields` — never fabricated.

## Adapter (project into, do not fork)

| Surface | Role |
| --- | --- |
| `formal/proof_trace.py` | Projection + hash + match/replay + mutation detectors |
| `runtime/telemetry/replay_bundle.py` | Source identity / choices / domain |
| `models/decode_stats.py` | Work counters + mechanism activation |
| `formal/event_trace.py` | Cost projection (`DecodeUnitWorkModel`) |
| `evals/semantic_failure.py` | Optional verifier witness refs |

Schema: `canonical_proof_trace/v1`. Every sealed trace sets
`claims_abstract_refinement=false` and `kern11_deferred=true`.

## Acceptance

- Re-projecting the same evidence yields the same `trace_hash` (replay match).
- Reordered, omitted, or stale-state mutations fail integrity / match.
- Refinement cannot be asserted by mere existence of a trace.
- No duplicate runtime recorder is introduced.

## Tests

`tests/test_formal/test_proof_trace.py` +
`resources/formal/proof_trace_fixtures.v1.json`.
