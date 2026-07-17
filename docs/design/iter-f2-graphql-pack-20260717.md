# F2 — GraphQL DSL pack (2026-07-17)

Fixture-grade implementation for Track F2 (Linear SLM-43) — the second real
DSL pack under the F1 contract and the first schema-native backend. Code:
[`src/slm_training/dsl/packs/graphql.py`](../../src/slm_training/dsl/packs/graphql.py),
resolving the shared `graphql` grammar backend via `get_backend("graphql")`.
Not a ship claim.

> **Reconciliation note (PR #275 merge, 2026-07-17):** origin/main already ships
> the canonical F2 GraphQL owner as `dsl/pack.py`'s `graphql` pack plus the
> Node-bridge [`graphql_js.py`](../../src/slm_training/dsl/grammar/backends/graphql_js.py)
> backend. #275's parallel `graphql_query.py` backend (pure `graphql-core`) was
> removed during the merge; `graphql_js` is the single registered `graphql`
> backend. #275's `dsl/packs/` pack framework is retained because Track G4's
> reasoning bench depends on its `arith-sketch` pack — see the F1 follow-up
> caveat in [`dsl-pack-contract.md`](dsl-pack-contract.md). The `dsl.packs`
> GraphQL pack now runs against the canonical `graphql_js` backend.

## What was built

- **Schema-native backend** — `GraphQLQueryBackend` implements the
  `GrammarBackend` Protocol directly on graphql-core (parse / print_ast /
  validate), no `.lark` grammar. `component_names()` returns the schema's
  type names and `library_schema()` the per-type field map, so the
  **introspection schema literally is the symbol table** — the point the
  program has wanted a real test for (OpenUI 0.2.x has almost no binding
  surface; this is the strongest C1/C2 scope substrate available).
- **Schema-aware oracle** — the pack's `validity_oracle` runs
  `graphql.validate` against a committed SDL fixture
  (`resources/graphql/shop_schema.graphql`). "Valid" means schema-correct:
  querying a non-existent field or passing a wrong argument name is
  rejected, not just malformed syntax. Verified in tests.
- **Real canonicalizer** — `print_ast(parse(x))` is the reference
  implementation's normal form (idempotent); unlike the toy/arith identity
  canonicalizers this genuinely collapses whitespace/formatting variants
  (`canonical_equal` holds across reformatting).
- **Deterministic corpus generator** — 5 query templates over the fixture
  schema (nested selections, arguments, enums), every emitted query
  self-checked through the oracle so no schema-invalid training row can
  escape.
- **Optional dependency** — graphql-core added as the `graphql` extra in
  `pyproject.toml`; the backend's `available()` gates all F2 tests exactly
  like `bridge_available()` gates the OpenUI bridge tests, so a venv without
  the extra skips cleanly rather than failing.

## Verification

- `tests/test_dsl/test_graphql_pack.py`: schema-is-symbol-table (type names
  + field map come from the SDL, not a hardcoded list), oracle is
  schema-aware not just syntactic (unknown field / wrong arg / malformed all
  reject), canonicalizer is a real idempotent normal form across
  reformatting, generator output is schema-valid and deterministic.
- The parametrized pack-contract invariant test now covers `graphql` too
  (valid corpus, scope-clean, idempotent canonical form), skipping when the
  optional dependency is absent.
- Grammar-backend suite green with the sixth backend registered;
  `repo_policy`, `ruff`, `git diff --check` clean.

## Honesty and limits

- **Query subset only**: the pack covers GraphQL *queries* against a fixed
  fixture schema. Mutations, subscriptions, fragments, directives, and
  multi-schema corpora are out of scope for this issue.
- **No experiment matrix / trained model yet**: F2 lands the pack
  (grammar/backend + schema-aware oracle + generator + contract tests). The
  quality-matrix rows, a scratch-trained GraphQL model, and constrained
  decode over the schema are the in-flight follow-ups (they ride the F1
  interface with no new plumbing, per the pack contract).
- **graphql-core, not graphql-js**: the design doc named graphql-js;
  graphql-core (the reference-tracking Python port) gives the same
  schema-aware validation with zero subprocess/bridge machinery. Byte
  parity with graphql-js is an explicit non-goal, recorded in the pack
  `notes` and `grammar-backends.md`.
