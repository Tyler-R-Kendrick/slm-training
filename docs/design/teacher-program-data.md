# Teacher program data

SLM-266 owns the future frontier-teacher prompt-plus-program corpus. The
canonical admission owner is
`src/slm_training/harnesses/experiments/teacher_programs.py`; it has no
provider client and therefore cannot accidentally spend budget during a data
build or training run.

## Frozen request and generation contracts

`TeacherProgramRequestV1` contains only source intent: coverage-gap identifiers
from the outcome-blind SLM-265 manifest, allowed public components, output
kind, protected-exclusion manifest hash, template hash, and seed. It must not
contain an OpenUI target, protected prompt, compiler token, binder index, or
evaluation outcome.

`TeacherProgramGenerationManifestV1` pins provider/model/revision, request
identifiers, protected-exclusion hash, and positive dollar/input/output hard
caps. Provider execution additionally requires the pinned input/output price
schedule and a locked campaign-manifest SHA. It preflights every request's
declared worst-case token/dollar use before the provider call.
`python -m scripts.build_teacher_program_generation_manifest` is the no-spend
writer for this locked manifest; it does not contact a provider.

`python -m scripts.generate_teacher_programs` is describe-only by default. With
`--execute`, it uses either an explicitly configured OpenAI-compatible endpoint
or `--local-transformers`, which loads the manifest's pinned model and revision
with Transformers `local_files_only=True` (the existing `hf` optional extra).
The local path never downloads a model or calls hosted inference; it records
device, dtype, model revision, and actual token usage in the same
content-addressed raw request/response/error archive plus hash-chained attempt events under
`outputs/autoresearch/<campaign>/`. Missing usage, price, or campaign lock
fails closed; no human rating is involved. Execution additionally loads the
declared `ExperimentCampaignV1`, locks it in that archive before the first
outcome event, and requires its digest to equal the generation manifest's
`campaign_manifest_sha256`.

Local transport supplies an explicit all-ones attention mask for the generated
prompt. Any process interruption records an `interrupted` experiment-finished
event; interrupted work is diagnostic only and cannot be counted as corpus
evidence. Every raw attempt and completion event is also bound to the same
generation manifest's experiment ID.

## Admission modes

`python -m scripts.admit_teacher_programs` consumes archived response records,
never contacts a provider, parses exactly one declared program payload without
repair, constructs `ProgramSpec`, and calls the shared verifier and protected
split leakage checker.

| Mode | Required policy | Deduplication | Eligible for canonical training |
| --- | --- | --- | --- |
| `deep_verified` | automated G0-G4, G7-G8, G10-G11, plus G5/G6 when requested; every required gate must be `pass`; admitted as Silver unless separately audited | one deterministic canonical-root representative | Yes |
| `parse_only` | G0-G1 only; later failures are retained | exact prompt/program pair only | No — controlled research only |
| `no_canonical_dedup` | same deep policy | exact prompt/program pair only | Controlled comparison only |

The generic verifier normally marks teacher rows Bronze until a human audit.
SLM-266 keeps that raw verifier result but derives an admission Silver tier after
the automated deep policy clears. G12 is retained as optional evidence, never a
human-rating gate. G11 remains the required independent automated judge; deep
admission requires its persisted candidate, cross-provider/model-family,
raw-artifact, prompt-hash, and program-hash evidence — a bare boolean cannot
clear the gate. Protected overlap is rejected before mode-specific
materialization.

`python -m scripts.materialize_teacher_programs` accepts only a
`deep_verified` admission result and produces `outputs/data/train/<id>/` with
`records.jsonl`, rejected rows, and a `DataStore` manifest. It carries a
required immutable raw archive URI plus manifest SHA; large provider I/O is not
copied into Git. The ordinary strict train-data builder and explicit
`DataStore.publish` remain the only canonical curation/publication path.

## SLM-296 normalization and hands-off boundary — 2026-07-25

SLM-296 adds `candidate_from_local_teacher_attempt` to the existing SLM-266
owner. It converts one content-addressed, `local_files_only` archive attempt
into a `TeacherProgramCandidate` without inventing a trust verdict. Deep
admission still requires cross-provider/family G11 evidence, now bound to the
exact raw artifact hash. G12/human audit is recorded-only and never admits or
promotes a row.

Focused local regression evidence: 15 `test_teacher_programs` cases passed,
including raw-artifact-bound hands-off deep admission. This is harness wiring,
not a corpus, train, checkpoint, or ship evaluation. The existing bounded CPU
yield disposition below remains the only measured local result: zero
deep-verified roots. Therefore SLM-296 does not claim 500 accepted records or
any grammar-off/meaning improvement; no HF, remote replay, or human-rating gate
was used.

## Current disposition — 2026-07-24

The one-request local CPU screening is recorded in
[`iter-slm266-local-qwen-screening-20260724.json`](iter-slm266-local-qwen-screening-20260724.json).
Qwen loaded from four cached local shards, but the 96-token generation did not
finish before the three-minute cap; it produced zero completed requests. The
archive was closed as interrupted and is diagnostic only: it is not a teacher
generation result, accepted corpus, admission/judge run, training result, or
ship claim. The required 10k corpus, three-seed factorial, and 100k rung remain
**unrun**; the 100k rung is explicitly deferred.

The follow-up [eight-token probe](iter-slm266-local-qwen-8token-probe-20260724.json)
did complete locally: 56 input tokens plus 8 output tokens in 95.65 seconds at
zero provider cost. Its response was a truncated fenced `<Panel>` fragment, so
it is transport evidence only and was not parsed, admitted, or materialized.

The corresponding [local yield disposition](iter-slm266-local-yield-disposition-20260724.json)
is `budget_blocked` for this exact CPU/model configuration: the only completed
request yielded zero accepted roots, while the 96-token request hit the
mandatory 170-second interrupt point. The 95.65-second completed-request
measurement permits at most one such request per capped command; even an
optimistic 10,000-request attempt sequence would take at least 11.07 continuous
days before parsing, deep verification, independent judging, deduplication, or
retries. This is a request-attempt lower bound, not a valid-program throughput
estimate. The 100k corpus remains unrun by direction; its 110.71-day
request-attempt lower bound is a bottleneck record, not an authorized run.

SLM-266 remains open: no accepted 10k principal corpus, matched controls, or
three-seed factorial exists. The disposition applies only to this local CPU
Qwen configuration and does not weaken corpus-admission, training, or ship
gates.
