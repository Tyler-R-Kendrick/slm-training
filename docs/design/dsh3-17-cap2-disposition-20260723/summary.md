# DSH3-17 CAP2 capability disposition (SLM-385)

Date: 2026-07-23
Status: CERT_CAP2 rejected; DSH4 action distillation closed
Scope: terminal evidence disposition; no model, checkpoint, or ship claim

## Decision

No action representation causally improved held-out CAP2 behavior; CAP1 retention is unavailable.
The compiler-owned operator contracts remain useful correctness infrastructure,
but no learned operator representation earned a capability certificate.

## Capability ledger

| Capability | Verdict | Implemented benefit | Evidence |
| --- | --- | ---: | --- |
| `symbolic_transform` | `contract_only` | false | SLM-381.cap2_operator_v1 |
| `nl_transform` | `unavailable` | false | none |
| `discrete_token_action` | `rejected` | false | SLM-382.E803 |
| `hierarchical_head` | `unrun_conditional` | false | none |
| `topology_application` | `unrun_conditional` | false | none |
| `bounded_merge` | `contract_only` | false | SLM-381.cap2_operator_v1 |
| `efficiency` | `unavailable` | false | SLM-382.E803 |

Symbolic transformation and bounded merge are `contract_only`: exact fixture
generation/replay exists, but no learned benefit passed. E803 rejects the
discrete token action. NL is unavailable without CERT_CAP1. The hierarchical
head and topology application remain unrun conditionals because their
prerequisite failed. No exact-hardware matched-quality efficiency evidence
exists.

## Certificate and downstream gate

- `CERT_CAP2`: **not issued**
- DSH4 action distillation: **closed**
- checkpoint/model-card roster change: **none**
- production/ship claim: **none**

## Evidence identities

### `SLM-381.cap2_operator_v1`

- class: `fixture_contract`
- code: `8a29de4b81da07393ec3acb3b906376baa593145`
- checkpoint: `None`
- data: `{"operator_corpus_fingerprint": "5ee0d27141a3fa72be35bedbdec347f97f513c0e7af672ca4be580e5b982682e", "source_records_fingerprint": "f18b2fa1d9e271fcb8789c766cbb3717262353d1bbddfd37c5a1b85bca16a00e"}`
- suite: `{"suite_hash": "16f210786bac7fd5f5edb64d13888c3cc7d634330a81b5065150e7a41fcb1d4d", "suite_n": 20, "suite_version": "cap2_operator_v1"}`
- config: `{"matrix_set": "cap2_operator_v1", "thresholds": {"accepted_legal_action_mass_min": 0.9, "case_wilson_lower_min": 0.75, "dimension_wilson_lower_min": 0.2}}`
- hardware: `{"device": "cpu", "efficiency_claim": false, "exact_hardware": null}`

### `SLM-382.E803`

- class: `bounded_matched_negative`
- code: `5cd5b8b6222b5a99c0bebdd43775f2aefb494165`
- checkpoint: `None`
- data: `{"held_out": "37b1fefce66c216ec287286be42c3fc4ce2fe8d018d300779858fbc9dfe4a1a1", "train": "2c963a3d014dde28599eb4e080cff99e8f035b7e13c0a3f5e6e089dd891ec9e0"}`
- suite: `{"held_out_n": 4, "suite": "CAP2 held-out operator decisions"}`
- config: `{"learning_rate": 0.03, "parameter_count": 34913, "seeds": [11, 29, 47], "steps_per_arm": 8}`
- hardware: `{"backend": "hashed_token_scorer", "device": "cpu", "efficiency_claim": false, "exact_hardware": null}`

## Integrity result

AgentV passed 4/4 cases with mean 1.0 and 0 execution errors.
No experiment was rerun for this disposition; it consumes the immutable
SLM-381 and E803 reports and preserves their positive, negative, unavailable,
and unrun boundaries.
