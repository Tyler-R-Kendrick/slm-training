# INTEG-08 — Corpus-local dominance and skip-decision evidence (SLM-562)

## Claim

When a treatment is **proved output-equivalent** to baseline on a locked corpus
and **no cheaper** under the declared cost model, emit a durable
corpus-local dominance evidence record and skip re-spend under equivalent knob
signatures. Results do not generalize past the exact
baseline/treatment/corpus/version/model/cost identity.

## Adapter (extend owners; no parallel store)

| Surface | Role |
| --- | --- |
| `autoresearch/preflight/mechanism_dominance.py` | Evidence + admission + discovered `CHECK` |
| KERN-08 `formal/mechanism_trigger.py` | Sole `DominanceCertificateV1` / no-effect authority |
| `autoresearch/hillclimb.ExhaustedKnobLedger` | Durable skip under `knob_signature_sha256` |
| `harnesses/experiments/mechanism_disposition_report.py` | Disposition projection (`feed_dominance_to_disposition`) |

Schemas: `corpus_local_dominance_evidence/v1`,
`mechanism_dominance_admission/v1`. Skip maps to preflight `block`;
non-dominated cases map to `pass`.

## Disposition rules

| Evidence | Disposition |
| --- | --- |
| Complete scan ∧ theorem-backed no-effect ∧ all rows `enabled_cost ≥ baseline_cost` ∧ dominance cert emits | `skip_dominated` |
| Complete scan ∧ no-effect ∧ any row strictly cheaper | `no_effect_cheaper` |
| Unknown/incomplete equivalence, costs, scan, present trigger, unsafe mechanism | `unknown` |

## Identity binding

Evidence seals `baseline_id`, `treatment_id`, `corpus_id`, `corpus_version`,
`model_id`, `cost_model`, `mechanism_id`, per-example costs, aggregate relation,
output-equivalence proof digest, optional empirical remainder, and certificate
digests. Replay reconstructs the disposition from `content_sha256`. Mutating
any identity field fails `scope_matches_evidence` / integrity and yields
`unknown` on replay.

## Acceptance

- Replay reconstructs `skip_dominated` / `no_effect_cheaper` / `unknown`.
- Corpus, model, or cost-model identity mutation invalidates the certificate.
- Dominated treatments record into `ExhaustedKnobLedger` with reason
  `corpus_local_dominance`; cheaper no-effect does not.

## Tests

`tests/test_autoresearch/test_mechanism_dominance.py`.
