# The DSL pack contract (F1, SLM-34)

A **DSL pack** is the unit of DSL pluggability for this stack:

```
DslPack = {
  grammar            — backend id in dsl/grammar/backends (parse / validate /
                       serialize / stream-check / token surfaces)
  canonicalize       — semantics-preserving normal form (+ fingerprint)
  validity oracle    — the authority that decides legality (never the model)
  corpus generator   — typed-AST generator producing training records
  scope rules        — reference representations + who enforces scope legality
  placeholder policy — content routing (identity semantics out of model scope)
  contract id        — content-derived language-surface hash (dataset stamping)
}
```

Code: [`src/slm_training/dsl/pack.py`](../../src/slm_training/dsl/pack.py)
(contract + registry), [`src/slm_training/dsl/packs/openui.py`](../../src/slm_training/dsl/packs/openui.py)
(first instance). Resolution mirrors the grammar switch: explicit id >
`SLM_DSL_PACK` > `SLM_GRAMMAR_DSL` > `openui`.

## Why a pack, not just a grammar backend

The `GrammarBackend` protocol already made *parsing* pluggable, but the
training loop's other DSL couplings were implicit OpenUI imports scattered
across the codebase: `canonicalize.py` (D2 normal form), `parser.validate`
(official lang-core oracle), `scope_corpus.build_scope_corpus` (typed-AST
generator), `placeholders.py` (routing policy), `language_contract.py`
(dataset stamping). F2 (GraphQL), F3 (patterns DSL), F4 (nomenclatures), and
G3 (latent-DSL generator) each need all seven members, so the contract makes
the implicit bundle explicit — a new DSL is a `register_pack(...)` call whose
members satisfy the same shapes, not a fork of the training stack.

Program-level invariants the contract encodes:

- **The oracle decides, the model proposes.** `validity_oracle` raises on
  illegal source; syntax legality is externalized (the E226 honest-compiler
  policy, and C1's verifier-enforced reference legality in `scope_rules`).
- **Content is routed, not memorized.** `placeholder_policy` is the C4
  "names disappear" defense: identity semantics stay out of the model's
  scope by construction.
- **Datasets are stamped.** `contract_id` binds corpora to the exact language
  surface, so silent grammar/library drift breaks loudly.

## Layering

Corpus generation lives in `harnesses/train_data` — *above* `dsl/` — so packs
hold lazy `module:attr` provider strings resolved on first use
(`corpus_generator()`), never direct imports. Everything else is a plain
callable on strings.

## Deliberately not moved

SLM-34 said "mostly moves"; the review of the tree said otherwise. The
component owners (`dsl/canonicalize.py`, `dsl/parser.py`,
`dsl/placeholders.py`, `dsl/language_contract.py`,
`harnesses/train_data/scope_corpus.py`) already have canonical, tested,
heavily-imported paths. Relocating them under `dsl/packs/openui/` would churn
every import site and conflict with in-flight branches for zero behavioral
gain, against `docs/repository-organization.md` ("extend the existing owner").
The pack wires the owners by reference; a future physical consolidation, if
ever wanted, is mechanical (`git mv` + provider-string updates) and invisible
to pack consumers.

## The second instance: GraphQL (F2, SLM-43)

[`dsl/packs/graphql.py`](../../src/slm_training/dsl/packs/graphql.py) proves
the contract generalizes:

- **Oracle**: the official `graphql` package (graphql-js) behind a JSON-over-
  stdio sidecar ([`src/apps/graphql_bridge`](../../src/apps/graphql_bridge))
  mirroring the OpenUI lang-core bridge. `validate` = parse + full schema
  validation.
- **Scope rules**: *the schema IS the symbol table* — every selected field
  must exist on its parent schema type, enforced by graphql-js, never
  learned. This is the strongest real-world exercise of the externalized-
  scope principle (C1/C2), since OpenUI 0.2.x has little binding surface.
- **Canonicalizer**: `print(parse(x))` — graphql-js's printer is a stable,
  idempotent normal form.
- **Placeholder policy**: variables (`$name`) are the routed-content channel;
  the typed-AST generator emits required arguments only through variables.
- **Corpus generator**
  ([`harnesses/train_data/graphql_corpus.py`](../../src/slm_training/harnesses/train_data/graphql_corpus.py)):
  walks the schema symbol table and emits one operation per Query root field
  with depth-bounded selection sets — every output passes the pack's own
  oracle by construction (enforced in `tests/test_dsl/test_graphql_pack.py`).
- **Contract id**: offline hash of the graphql package version + the fixture
  schema ([`resources/graphql/demo_schema.graphql`](../../src/slm_training/resources/graphql/demo_schema.graphql)).

Deferred (recorded, not claimed): running the OpenUI quality matrices/gates
over GraphQL corpora requires a GraphQL-aware output tokenizer and suite
definitions — that is the training half of F2 and lands separately; this
change is the pack foundation (oracle, scope, generator, canonical form).

## What F2+ must implement

1. A `GrammarBackend` (register in `dsl/grammar/backends`) whose
   `validate`/`serialize` round-trip is the official oracle for that DSL —
   for GraphQL: graphql-js via a bridge, the introspection schema as the
   symbol table (the F2 scope-rules instance).
2. A canonicalizer with the idempotence + fingerprint properties
   (`canonicalize(canonicalize(x)) == canonicalize(x)`).
3. A typed-AST corpus generator returning `ExampleRecord`s whose outputs pass
   the pack's own oracle (enforced by the F1 end-to-end test shape).
4. Scope rules naming the reference encodings and the verifier that enforces
   legality.
5. A placeholder policy — what content is routed out of model scope.
6. A `contract_id` derived from the language surface's actual content.

## Verification (F1 gate)

`tests/test_dsl/test_pack_contract.py`: registry/env resolution; member
identity with the existing owners; canonicalizer idempotence + stable
fingerprint; oracle rejects garbage; and the end-to-end fixture —
**generate** (scope corpus from one root program) → every document record
passes the pack's oracle → **train scratch** (tiny CPU TwoTower, real
gradient steps, finite loss) → **eval** (constrained decode, certified back
through the pack's oracle/canonicalizer). Existing suites untouched — the
contract adds wiring, no owner changed.
