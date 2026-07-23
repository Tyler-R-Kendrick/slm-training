# DSH0-04 content-addressed capability artifacts (SLM-348)

**Disposition:** adopt `capability_artifacts/v1` as the immutable artifact
contract for staged answers, questions, accepted QA sets, derivations,
validations, and capability results. This is a contract fixture, not a data
build, train/eval/benchmark, checkpoint, capability certificate, or ship claim.

Machine-readable evidence:
[`dsh0-04-capability-artifacts-20260723.json`](dsh0-04-capability-artifacts-20260723.json).

## Identity boundary

Semantic content and process activity have deliberately separate identities.

| Record | Content/process identity |
| --- | --- |
| `AnswerArtifactV1` | canonical AST, marker table, grammar start/category, complexity |
| `QuestionArtifactV1` | question digest, marker table, grammar start/category, complexity |
| `QAPairArtifactV1` | question plus sorted accepted-answer set and equivalence relation |
| `DerivationActivityV1` | invocation, timestamp, sources/outputs, process/config/code versions, seed, optional LLM/teacher trace |
| `ValidationReportV1` | invocation, sources, validator process, rejection codes, compiler coverage |
| `CapabilityCertificateV1` | capability, exact plan, QA/validation evidence, gate process, disposition |

Answer surface digests, family/split/parent lineage, and equivalence relations
remain immutable record fields but do not perturb canonical AST semantic
identity. Timestamps and invocation IDs exist only on activities/reports.
Thus equivalent canonical answers can be reused while two executions retain
distinct activity IDs.

Accepted answer membership has its own `accepted_set_id`. The optional
canonical preference must belong to that set, but changing preference does not
rewrite accepted semantics.

## Publication boundary

`require_publishable` resolves every parent, equivalence, question/answer,
derivation input/output, validation, and certificate evidence ID against the
publication batch or explicitly known external store IDs. Duplicate artifact
IDs or any unresolved edge reject publication.

All process-backed records require non-empty process IDs and versions plus
exact config/code SHA-256 identities. LLM use records provider, model, prompt,
and response digests. Teacher traces explicitly distinguish `exact` from
`approximate` and pin teacher/version/trace identity.

Accepted validations require complete compiler coverage and no rejection
codes. Rejected validations and failed certificates require typed rejection
codes. No missing evidence becomes a passing result.

## Serialization and migration

Canonical JSON sorts relation sets and keys. Strict loading reconstructs each
typed artifact, recomputes its recorded identity, and compares the complete
canonical payload. Unknown fields, altered IDs, unsupported artifact types,
and schema versions other than `capability_artifacts/v1` fail closed.

Eight focused tests cover semantic/activity separation, QA preference
independence, process/LLM/teacher provenance, validation/certificate gates,
whole-graph publication, missing provenance, deterministic JSON round trips,
and migration/identity rejection.

## Next disposition

The artifact-store/runner follow-up may persist these records by their
computed IDs. This contract does not itself publish artifacts or grant a
capability certificate.
