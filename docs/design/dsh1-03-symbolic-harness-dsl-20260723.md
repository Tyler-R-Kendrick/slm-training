# DSH1-03 symbolic Harness DSL (SLM-355)

**Decision:** supported at the deterministic contract-fixture level. CAP0
identity and canonicalization intent now uses a closed, versioned Harness DSL
instead of English instructions. The parser resolves exactly one reserved
operation and one typed payload, preserves exact pack and grammar-category
identity, and rejects malformed framing or embedded fragments before shared
model-data construction.

Machine-readable evidence:
[`dsh1-03-symbolic-harness-dsl-20260723.json`](dsh1-03-symbolic-harness-dsl-20260723.json).

## Contract

`harness_dsl/v1` reserves `IDENTITY`, `CANONICALIZE`, `COMPLETE_SUFFIX`, and
`COMPOSE`, with document, statement, expression, lexical, and node payloads.
Canonical framing contains only reserved fields, exact target-pack and
grammar-category symbols, optional lowercase SHA-256 artifact references,
declared runtime markers, and one byte-exact embedded target fragment. The
checked-in grammar and schema version produce a stable SHA-256 fingerprint.

The outer parser is independent of OpenUI. Once framing is closed, it resolves
the selected `DslPack`, requires that pack's typed `fragment_parser`, and
applies the pack-scoped symbolic-surface policy. OpenUI-specific fragment
validation stays inside its pack adapter. Comments, open prose strings,
undeclared identifiers or runtime refs, unknown fields, delimiter injection,
duplicate refs/markers, invalid categories, and trailing text fail closed.

## CAP0 conversion and boundary

Existing `scope_identity_*` rows map to `IDENTITY`; `scope_canonical_*` rows
map to `CANONICALIZE`. Conversion preserves the complete `ExampleRecord` and
lineage metadata, adds the schema/fingerprint/operation/pack/type/category
envelope, and keeps preference prompts byte-identical to their records and
identity twins.

`scope_repair_*` and `lexical_typed_map` remain unchanged because their
semantics do not match the reserved operations. They fail closed if passed to
the CAP0 conversion helper. `COMPLETE_SUFFIX` and `COMPOSE` are available for
later compiler-prefix and artifact-composition producers; this change does not
relabel existing examples.

Shared train/eval record loading parses any Harness-marked prompt and compares
it with its metadata before enforcing the output contract. Malformed or
metadata-divergent prompts therefore fail before model input construction.
Prompt-contract projection skips Harness rows so it cannot append English.

## Verification and claim limits

The focused Harness DSL, scope-corpus, and model-data boundary suites pass 46
tests. They cover all 20 operation/type combinations, canonical round trip,
pack/category/artifact identity, grammar fingerprinting, symbolic-surface
rejections, declared-marker closure, CAP0 conversion field preservation,
unsupported-family refusal, preference prompt identity, and malformed-loader
rejection.

The repository changed-file hook additionally passes 1,431 tests with eight
skips and 12 policy deselections. Ruff, compileall, version stamps, repository
policy, and diff checks pass.

This is deterministic repository-contract evidence. No corpus build, train,
model evaluation, benchmark, checkpoint, AgentEvals publication, capability
certificate, or ship claim was produced. Reserved `COMPLETE_SUFFIX` and
`COMPOSE` syntax is not evidence that their future producers or learned
capabilities exist.

## Research lineage

This is a repository protocol boundary, not a new learned mechanism and not a
paper reproduction. It operationalizes the staged-harness requirement that
CAP0 task intent be structural, typed, versioned, and pack validated.
