# DSH1-04 marker tables with alpha-renaming (SLM-356)

**Decision:** supported at the deterministic contract-fixture level. Runtime
symbols from the request contract now materialize into versioned marker tables
whose model-facing rows carry only opaque request-local surfaces plus declared
typed authority (role/type/scope/signature). The logical-to-surface mapping
lives exclusively in a sidecar provenance record.

Machine-readable evidence:
[`iter-slm356-marker-tables-20260725.json`](iter-slm356-marker-tables-20260725.json).

## Contract

`marker_table/v1` (`src/slm_training/harnesses/train_data/marker_tables.py`)
projects `GenerationRequest.effective_runtime_symbols()` onto four marker
families — binder, external entity, state reference, fresh binder — using the
existing `RuntimeSymbol` surface codecs (`:` external, `$` state) so tables
stay compatible with the V8 dynamic-vocabulary path
(`SymbolTable.from_runtime_symbols`). Only inference-visible facts
(`semantic_type`, `semantic_role`, `scope`, `signature`) are copied onto rows;
`description` and the logical surface never leave the provenance sidecar
(`marker_table_provenance/v1`).

## Split-level permutation

Ordinal assignment is a deranged permutation seeded by
SHA-256(policy | master_seed | root_family | split), so train and held-out
splits get **independent** surfaces. Surface namespaces are salted per
(family, split) seed, which makes train/eval surface sets disjoint, and the
split itself is enforced through `RootFamilySplitPolicyV1` inheritance.
`permute_marker_surfaces` alpha-renames a table under a fresh seed; the
surface-free canonical payload digest (`canonical_marker_payload`) is
invariant, so canonical AST identity survives renaming.

## Audits (fail closed)

`audit_marker_table` / `require_clean` / `resolve_marker` emit blocking
findings and raise on: exact-copy surfaces (model surface equals its logical
surface), semantic surface dependence (marker embeds a logical name — the
corpus-publication stop rule), alpha mismatch (canonical identity changed
under renaming), unknown markers, duplicate marker use, out-of-range markers,
and ordinal-role leaks (declaration-order index mapping to an identical
ordinal). Role is declared authority via the surface codec prefix; the guard
is that role is **not** recoverable from the ordinal stream, checked by
`assert_role_not_ordinal_recoverable` over held-out fixtures.

## Results

- `tests/test_harnesses/train_data/test_marker_tables.py`: 13 passed
  (determinism, disjoint split surfaces, alpha invariance, provenance-only
  logical mapping, unknown/duplicate/out-of-range/exact-copy fail-closed,
  role-not-recoverable fixture, split-policy inheritance).
- Runtime-symbol / V8 and train-data suites: 150 passed, 5 pre-existing
  failures on this branch unrelated to this change (4x
  `test_staged_materialization.py`, plus the dsh0-02 symbolic-surface evidence
  hash drift vs committed `src/slm_training/dsl/pack.py`).
- `python -m scripts.verify_version_stamps --check`, `repo_policy`, ruff, and
  `git diff --check`: passed.

Registry: new component `harness.experiments.slm356_marker_tables` v1;
`harness.train_data` bumped to v20 (new module under its watched directory).

Claim limits: fixture-scale contract evidence only — no corpus publication,
no model evaluation, no ship-gate claim.
