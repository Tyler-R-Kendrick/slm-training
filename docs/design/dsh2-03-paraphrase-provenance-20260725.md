# DSH2-03 provenance-complete NL paraphrase provider + cache (SLM-364)

**Decision:** supported at the contract-fixture evidence level.
`paraphrase_provenance.py` defines a provider-neutral, content-addressed
request/response contract for turning a `SemanticFrameV1` + `CAP1SchemaV1`
(DSH2-02, SLM-363) into prompt-side natural language: every admitted
paraphrase resolves exact provider/model/prompt/schema/frame provenance, the
same request identity always reuses one cached artifact, and a raw response
only reaches a usable "question artifact" after it passes quarantine
validation. No real LLM was called; this is fixture/contract-level evidence.

Machine-readable evidence:
[`dsh2-03-paraphrase-provenance-20260725.json`](dsh2-03-paraphrase-provenance-20260725.json).

## SLM-343 / "StructuredDslSchemaV1" interpretation

The issue text names a class, `StructuredDslSchemaV1`, that does not exist
anywhere in this repository. SLM-343 ("CAP1-GEN-01: define the generic
NL↔DSL grounding contract and capability certificate") — the issue that would
define it — remains a design-doc-only backlog issue with no merged code, the
same status recorded in `dsh2-02-semantic-frame-20260725.md` one issue prior
in this milestone. This module makes the identical substitution documented
there: it reuses `CAP1SchemaV1` (`slm_training.dsl.semantic_frame`) as the
closest already-merged equivalent of "the structured DSL schema" the issue
means, and does not invent a second parallel `StructuredDslSchemaV1` type.
`derive_semantic_frame` + `CAP1SchemaV1.from_pack()` are the two objects a
request is built from; there is nothing else to substitute.

## Placement

`src/slm_training/dsl/paraphrase_provenance.py` (not
`harnesses/train_data/`): it is a direct, same-milestone continuation of
`dsl/semantic_frame.py` — it imports `CAP1SchemaV1`/`SemanticFrameV1`
directly and produces nothing else. `harnesses/train_data/synth.py` already
establishes a `PromptSynthesizer` `Protocol` for *training-record* paraphrase
expansion, but that plugin operates on already-materialized `ExampleRecord`
rows post-synthesis; this module is upstream of that — the request/response
provenance and caching contract for one external-LLM call over a
schema+frame pair, independent of whether its eventual output ever becomes an
`ExampleRecord`. Keeping it beside `semantic_frame.py` avoids a new package
boundary for a two-file addition and keeps the DSH2 milestone's producer
modules in one place.

## Reused idioms (no shadow paths)

- **Content addressing:** `cache_key_for` and the frame-fingerprint hash both
  call `slm_training.harness_core.lineage.records.content_sha`/
  `canonical_json` — the same primitive `RunManifest.sha`,
  `DataSnapshot.sha`, and a dozen other harness modules already use. No new
  hashing scheme was introduced.
- **Protocol-based pluggability:** `ParaphraseProvider` is a `typing.Protocol`
  matching the house idiom already used for `PromptSynthesizer`
  (`harnesses/train_data/synth.py`), `GenerationBackend`/`TrainingBackend`
  (`harnesses/preference/remine_campaign.py`), and `GrammarBackend`
  (`dsl/grammar/backends/types.py`): a small structural interface plus one or
  more concrete, fully-implemented backends.
- **Frozen dataclasses + `to_dict()`:** every typed record
  (`SamplingParamsV1`, `ParaphraseRequestV1`, `ParaphraseResponseV1`,
  `QuarantinedResponseV1`, `QuestionArtifactV1`) follows the
  `dsl/schema.py` / `harness_core/lineage/records.py` house style — frozen
  dataclass, explicit `__post_init__` validation where it applies, explicit
  `to_dict()`.
- No repo-wide `SyncTransport`/`linear_sync.py` idempotent-transport pattern
  was found merged in this checkout to mirror (searched
  `src/slm_training/harnesses/` and repo-wide for `redact`/`SyncTransport`/
  `idempoten*`); the closest merged precedent for "provenance record with
  provider/model/digest/timestamp identity" is
  `harnesses/distill/trace_store.py`'s `DecodeTraceRecorder`/`TraceStore`
  (checkpoint SHA + decode-config hash + seed identity, append-only store).
  This module follows the same shape (typed identity fields, content-addressed
  key, one store per identity) rather than that append-only JSONL persistence,
  because "same request identity reuses one cached artifact" is exactly
  addressable-dict semantics and an in-memory `ReplayCache` satisfies every
  acceptance bullet without inventing a new disk format.

## Sendable payload: schema + frame only

`build_prompt_payload(schema, frame)` returns exactly `{"schema": ..., "frame":
...}`. The frame projection deliberately drops
`SemanticFrameV1.canonical_source` — the literal canonical DSL target text —
because that is exactly the target-only content the issue says must never be
sent: an LLM that can read the literal DSL source is a copier, not a
paraphraser of declared facts, and letting it see DSL syntax risks the same
"prose exposes DSL/placeholder syntax" failure mode
`abstraction-house-style.md` rejects. Neither `CAP1SchemaV1` nor
`SemanticFrameV1` carries a split, label, or other eval/target-only field, so
no further field-level exclusion was needed — `test_paraphrase_provenance.py`
asserts this by content-scanning the serialized payload.

## Provenance record

`ParaphraseRequestV1` records provider id, model id/revision,
`schema_fingerprint` (`CAP1SchemaV1.fingerprint()`), `frame_fingerprint`
(`content_sha` of the frame projection), `system_digest`/`prompt_digest`
(sha256 of the exact system/prompt text sent — digest-only, never the raw
text, so the record itself can never carry more than the schema+frame
projection already declares sendable), `SamplingParamsV1`, optional `seed`,
`request_id`, and `created_at`. `ParaphraseResponseV1` records only
`response_digest` (sha256 of the redacted response text) and a timestamp —
the actual (redacted) text lives solely in the quarantine/promoted stores,
keyed by cache key, never inside the provenance record itself.

## Content-addressed cache key

`cache_key_for` hashes exactly `{provider_id, model_id, model_revision,
prompt_digest, system_digest, sampling.to_dict(), seed}` via `content_sha`.
`request_id` and every timestamp are excluded — they are incidental identity,
not reproducibility inputs — and a test proves two requests differing only in
`request_id`/timestamp resolve to the identical cache key while a request
differing in `seed` or sampling does not.

## Providers

- `DeterministicFixtureProviderV1` — fully offline, credential-free, no
  network: `generate()` reads the JSON-rendered prompt payload back out,
  picks one of three fixed templates by hashing
  `(system_text, prompt_text, sampling, seed)` with `content_sha`, and fills
  it from frame fact/node/effect counts. This is the provider every
  acceptance test exercises end-to-end.
- `ApiKeyStubProviderV1` — structurally wired to the shape of a real hosted
  API (provider id, model id/revision, `api_key`, `endpoint`), but its
  `generate()` raises `ProviderNetworkDisabledError` unless a `transport`
  callable is explicitly injected; production code has none wired, so a bare
  instance never performs a network call. Tests inject a plain Python
  callable (never a socket) as `transport` to exercise the full
  request/response/redaction contract without ever touching a network — this
  sandbox has none anyway, and the issue is explicit that no real hosted call
  is in scope.

## Quarantine → validation → promotion

Every raw response is redacted (`redact_text`, scrubbing every string
`provider.secrets()` declares) before it is wrapped in a
`QuarantinedResponseV1` and written to the `ReplayCache` — this happens
unconditionally, before validation. `validate_raw_response` then checks:
non-empty, not a verbatim echo of the structured prompt payload, under a
length cap, free of control characters, and free of any registered secret
string. Only a response that passes becomes a `QuestionArtifactV1` via
`ReplayCache.promote`, which itself refuses (`QuarantineIntegrityError`)
unless the quarantine entry's status is `"validated"`. A rejected entry is
never promoted, including on a replayed request under the same identity (the
rejection is itself cached and reused, so a permanently malformed
provider/schema/frame combination doesn't get silently retried into a
different verdict).

## Redaction

`ApiKeyStubProviderV1.__repr__`/`__str__` render the credential as the fixed
string `'[REDACTED]'` unconditionally. `redact_text` scrubs every declared
secret out of the raw response before it is digested, quarantined, or
promoted, so the persisted `response_digest` and stored text are already
post-redaction. The negative-control test constructs the stub with a fake
secret, wires a transport that deliberately echoes that secret into the raw
response text, runs the full `request_paraphrase` pipeline, and asserts the
secret string is absent from the serialized quarantine entry, the serialized
promoted artifact, the promoted `paraphrase_text`, and both `repr()`/`str()`
of the provider.

## Partial failure / retry / rate limit

`request_paraphrase(..., max_attempts=N)` retries `ProviderError` up to `N`
times; only the final successful raw response is redacted, quarantined, and
validated. On total failure (including a rate-limit error that outlasts
`max_attempts`) the call raises and the cache is left untouched — no
quarantine entry, no promoted artifact, nothing occupies that identity's slot
that a later successful retry could conflict with. `dry_run=True` builds the
request and cache key and returns without ever calling the provider or
touching the cache, for exercising identity/cache-key logic in isolation.

## Evidence

| Control | Result |
| --- | --- |
| sendable payload contains only `schema`+`frame` keys | pass |
| sendable payload excludes `canonical_source` / literal DSL text | pass |
| sendable payload has no split/gold/eval_only/target_kind field | pass |
| cache key ignores `request_id` and timestamps | pass |
| cache key changes with `seed` and sampling params | pass |
| `cache_key_for` matches the key computed during a live request | pass |
| same identity reuses one cached artifact, provider called once | pass |
| promoted artifact resolves exact provider/model/schema/frame/prompt provenance | pass |
| distinct frames yield distinct frame fingerprints and cache keys | pass |
| `dry_run=True` never calls the provider or writes the cache | pass |
| empty response stays quarantined, never promoted (repeat included) | pass |
| verbatim-echo response is rejected, never promoted | pass |
| `validate_raw_response` unit-level empty/echo/secret-leak checks | pass |
| `ReplayCache.promote` refuses without a validated quarantine entry | pass |
| total provider failure publishes nothing (cache stays empty) | pass |
| retry recovers and promotes only the final successful response | pass |
| rate limit outlasting `max_attempts` publishes nothing | pass |
| fake secret never leaks into quarantine/artifact serialization or repr/str | pass |
| API-key-shaped stub never calls "network" without an injected transport | pass |
| `SamplingParamsV1` rejects out-of-range temperature/top_p/max_tokens | pass |
| deterministic fixture provider is deterministic given identical inputs | pass |
| deterministic fixture provider output varies across seeds | pass |

The focused new-module suite passed 27 tests
(`tests/test_dsl/test_paraphrase_provenance.py`). A combined run with the
DSH2-02 module it depends on (`test_semantic_frame.py`) passed 44 (27 + 17).
Ruff, `ruff format --check`, and `scripts.verify_version_stamps --check`,
`scripts.repo_policy`, and `git diff --check` all passed on the touched
files (exact commands + output in the machine-readable evidence file).

This is deterministic fixture/contract-level evidence. No real LLM API was
called (no network access, no credentials, no cost); no corpus synthesis,
train, model eval, benchmark, checkpoint, AgentEvals publication, capability
certificate, or ship claim was produced.

## Honest limitations

- `ApiKeyStubProviderV1` is structurally wired (provider id, model
  id/revision, `api_key`, `endpoint`, an injectable `transport`) but is never
  invoked against a real hosted API anywhere in this change — no production
  code path constructs it with a real transport. It exists only to prove the
  contract is provider-neutral and to exercise the secret-redaction negative
  control without a network call.
- `ReplayCache` is in-memory only, scoped to one process/run. No disk
  persistence, no cross-run replay, and no distributed cache were built — the
  issue's acceptance bullets ("same request identity reuses one cached
  artifact") are testable and satisfied without one, and no established
  on-disk pattern for this exact shape was found to reuse (see "Reused
  idioms" above).
- `validate_raw_response`'s "matches expected shape" check is a generic
  well-formedness/echo/length/secret-leak gate, not a semantic-quality
  judge — it cannot catch a fluent-but-factually-wrong paraphrase. Semantic
  grounding of a promoted paraphrase against the frame's required/forbidden
  facts (e.g. via `semantic_frame.check_constraint`) is a natural next step
  but out of scope for this issue's provenance/caching contract.
- The retry loop is a plain bounded loop over `ProviderError`, not a backoff
  or jitter policy — sufficient to prove "retry cannot publish incomplete
  rows" without overbuilding a production rate-limit client this issue does
  not call for.
- This is a fixture/contract-level module only; nothing here was used to
  synthesize, train, or evaluate anything.

## Research lineage

The request/response identity fields (provider, model id/revision, digests,
sampling, seed, request id, timestamps) mirror the teacher/decode-provenance
shape already established in `harnesses/distill/trace_store.py`
(`DecodeTraceRecorder`/`TraceStore`: checkpoint SHA + decode-config hash +
seed identity) and the `RunManifest`/`DataSnapshot` content-addressed
provenance records in `harness_core/lineage/records.py`; this module adapts
that same "typed identity + content hash" discipline to one external
paraphrase call instead of a decode trajectory or a training run.
