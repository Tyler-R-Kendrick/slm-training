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
`--execute`, it uses an explicitly configured OpenAI-compatible endpoint and
stores content-addressed raw request/response/error artifacts plus hash-chained
attempt events under `outputs/autoresearch/<campaign>/`. Missing usage, price,
or campaign lock fails closed; no human rating is involved.

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
human-rating gate. G11 remains the required independent automated judge; its
family must differ from the generator. Protected overlap is rejected before
mode-specific materialization.

`python -m scripts.materialize_teacher_programs` accepts only a
`deep_verified` admission result and produces `outputs/data/train/<id>/` with
`records.jsonl`, rejected rows, and a `DataStore` manifest. It carries a
required immutable raw archive URI plus manifest SHA; large provider I/O is not
copied into Git. The ordinary strict train-data builder and explicit
`DataStore.publish` remain the only canonical curation/publication path.

## Current disposition — 2026-07-24

This is implementation and fixture evidence only, not a teacher generation,
accepted corpus, training result, or ship claim. There is no configured provider
credential, nonzero approved generation budget, raw archive, or independent
judge configuration in this checkout. Therefore the required 10k/100k corpora
and three-seed factorial remain **unrun**; the next execution must create a
locked manifest and durable raw archive before any provider spend.
