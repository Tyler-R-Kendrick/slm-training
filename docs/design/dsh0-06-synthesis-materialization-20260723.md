# DSH0-06 synthesis-plan materialization (SLM-350)

**Disposition:** adopt the plan-authoritative artifact-graph adapter into the
existing train-data pipeline. This is repository-contract validation with
temporary test fixtures, not a durable corpus build, train/eval/benchmark,
checkpoint, capability certificate, or ship claim.

Machine-readable evidence:
[`dsh0-06-synthesis-materialization-20260723.json`](dsh0-06-synthesis-materialization-20260723.json).

## One admission path

`--synthesis-plan` remains default-off and independent of `--curriculum`. When
present, the configured output directory must be one of the plan's pinned
destinations and already contain the append-only artifact graph. Question,
answer, and QA-pair nodes lower into the existing `ExampleRecord` and
`PreferencePair` contracts; no second loader, row schema, quality report, or
manifest was introduced.

Each accepted answer becomes one staged `ExampleRecord`. Its metadata pins the
plan ID/hash and typed question, answer, QA-pair, root-family, split-group, and
graph-node IDs. Canonical preference stays separate from accepted-answer
semantics: a preference pair ranks the explicitly selected canonical surface
over another accepted equivalent without relabeling that alternative as
incorrect.

## Fail-closed validation

Every staged target traverses the normal normalization, verification, quality,
test-structure firewall, decontamination, dedup, exposure, and publication
stack. Before normal admission, it additionally must pass:

- `SymbolicSurfacePolicyV1` under the plan's exact policy and runtime symbols;
- the active `DslPack` parse, complete static stream check, serialize/reparse,
  and idempotent canonical round trip;
- the existing synthetic-integrity checks, including production/choice codecs,
  slot contract, references, and request/target agreement;
- a DSL-native tokenizer encode/decode canonical round trip.

Malformed graph payloads, normalization failures, and staged-validation
failures remain in `rejected.jsonl` and create typed graph quarantine evidence.
Preference rows publish only when every referenced answer record survived the
full admission stack.

## Publication and determinism

The existing manifest now conditionally publishes the exact plan, artifact
graph path/schema/hash/node IDs, graph version stamp, accepted/rejected and
quarantine counts, preference count, and a deterministic `DATASET_CARD.md`.
The card and `records.jsonl` are byte-stable across resume/rebuild; ordinary
no-plan builds do not create these fields or files.

The live CAP0 fixture was refreshed for the behavior-changing train-data
component:

| Authority | Identity |
| --- | --- |
| plan | `dsh0-cap0-fixture` / `f0778bd88005687242a73626bdf8b4239750bbc9f6d10c06fae92a6240c9daa1` |
| pack | `openui` / `b37da286bdbcc41c3227f7ec6a379f78c9c62348be682cdbd7544c4feb77cb2d` |
| generator | `pack.corpus_generator` / `harness.train_data/v17` |
| validators | `pack.oracle` / `v17`; `symbolic_surface` / `v1` |
| graph | `artifact_graph_sidecar/v1` |

## Acceptance evidence

Focused tests pin the historical no-plan fixture `records.jsonl` digest, prove
two answer nodes become two fully validated records plus one canonical
preference, verify deterministic rebuild bytes, and prove invalid symbolic
content emits zero accepted rows with retained rejection and quarantine
evidence. The synthesis-plan state machine and artifact-graph traversal/
firewall fixtures remain green.

No production dataset was materialized for this contract change. Therefore
there is no new synthesis feedback decision, AgentV result, model-card entry,
checkpoint, capability certificate, or ship-gate conclusion.
