# DSH2-02 deterministic SemanticFrameV1 (SLM-363)

**Decision:** supported at the contract-fixture evidence level.
`derive_semantic_frame` reads a `DslPack`-declared CAP1 schema plus the
canonical OpenUI AST and returns a `SemanticFrameV1`: entities, roles,
relations, order, cardinality, closed-value alternatives, effects, and a
partition of facts into `required` / `optional` / `forbidden` / `unspecified`,
each carrying exact AST-path and schema-field provenance. It has no
free-form/LLM-guessed fallback: any component, prop, or AST shape the schema
does not declare raises `UnsupportedSemanticFactError` instead of being
guessed.

Machine-readable evidence:
[`dsh2-02-semantic-frame-20260725.json`](dsh2-02-semantic-frame-20260725.json).

## SLM-343 interpretation

SLM-343 ("CAP1-GEN-01: define the generic NL↔DSL grounding contract and
capability certificate") is a design-doc-only backlog issue with no merged
code, so this module imports nothing from it. It instead derives its own
minimal, explicit `CAP1SchemaV1` from artifacts that *are* merged: the same
`DslPack.prop_order` / `backend.component_names()` / `backend.content_props()`
slots the DSH1-01 `GrammarCapabilityAdapterV1` already fingerprints — no
parallel schema is invented. The interpretation adopted of SLM-343's stated
invariant ("prompt/context may contain natural language; the DSL target
contains only symbolic structure") is: every prop declared a *content prop*
holds free natural-language text; a placeholder token there is a required,
exactly-provenanced fact, while the natural-language text the placeholder
will eventually be filled with is `unspecified`. Everything else in the AST —
component types, structural prop values, order, cardinality, closed-value
literals, and state/query/mutation relations and effects — is symbolic and
must be derivable exactly.

Two further schema-level policy choices are documented, not derived from the
grammar, and are empty/off by default so they cannot silently invent facts:

- `closed_value_domains` lets a caller declare additional closed string/number
  domains beyond what this module always treats as closed by grammar
  construction (Python `bool` literals; the exact binary/unary operator token
  sets in `openui.lark`). A value outside a declared domain fails closed as a
  schema/AST mismatch rather than being accepted.
- `default_values` optionally marks a `(component, prop)` value as a schema
  default, downgrading a matching fact from `required` to `optional`. SLM-343
  defines no default table yet, so the shipped `CAP1SchemaV1.from_pack()`
  factory leaves this empty — every structural fact is `required` unless a
  caller explicitly opts a component/prop pair into `optional`.

`@Run` / `@Set` / `@Count` and the rest of OpenUI's builtin-call vocabulary
(`slm_training.dsl.openui_tokens.STRUCTURAL_TOKENS`) are **not** in the
schema's closed effect vocabulary: that list is documented there as a
decode-time token prior for constrained decoding, not a hard grammar/schema
authority, so promoting it to schema authority was out of scope here. Calls
to those builtins fail closed rather than being guessed a role.

## Frame contract

- `SemanticFrameNodeV1` — one AST node (element / ref / state-ref /
  runtime-ref / inline effect call / expression combinator), each with a
  deterministic pre-order id and its exact `ast_path`.
- `SemanticRoleV1` — a parent → child edge naming which prop/slot (and, for
  list-valued slots, which order index) the child fills.
- `SemanticRelationV1` / `SemanticEffectV1` — non-tree bindings: state reads,
  query reads, and action/mutation invocations (side-effecting) versus query
  reads (non-side-effecting).
- `SemanticFactV1` — one atomic fact: `category` (`required` / `optional` /
  `forbidden` / `unspecified`), `predicate`, `ast_path`, `schema_field`,
  `value`. Every fact — including every `required` and `forbidden` one —
  carries both provenance fields unconditionally.

`derive_semantic_frame(source, schema)` accepts raw or already-canonical DSL
text (canonicalized internally via the existing D2 canonicalizer, so two
sources denoting the same layout parse to byte-identical canonical ASTs) or
an already-validated `Program`. `SemanticFrameV1.equivalent_to` compares
schema fingerprint plus the four fact/role/relation/effect/node sets as
frozensets — deliberately ignoring incidental canonical-source-string
identity. `frame_conflicts` detects internally contradictory fact pairs
(same AST-path/schema-field slot asserted both required and forbidden, or
required to two different values); it is empty for every frame this module
derives and exists to check hand-assembled or mutated frames.

## Inverse frame-to-constraint and counterfactuals

`frame_to_constraint` turns a frame's `required`/`forbidden` facts into a
`FrameConstraintV1` of `requires`/`forbids` clauses (optional/unspecified
facts impose no constraint). `check_constraint` evaluates that constraint
against a *candidate* AST by deriving the candidate's own frame and comparing
fact values — it never emits DSL source text itself, satisfying the "without
generating DSL text" requirement for candidate/ambiguity evaluation.

`mutate_one_fact` flips exactly one `closed_value` required fact to its
schema-declared alternative (the only fact kind with an exact, non-guessed
alternative — its sibling `forbidden` facts already enumerate it) and returns
a `CounterfactualProofV1` from `verify_single_fact_change`, which recomputes
the asserted-fact diff between the two frames and raises unless exactly one
`(ast_path, schema_field, predicate)` slot changed. A frame with no
closed-domain fact raises `CounterfactualError` rather than guessing a
replacement for an open-domain literal (free text, an arbitrary number).

## Evidence

| Control | Result |
| --- | --- |
| every required/forbidden fact carries exact ast_path + schema_field | pass |
| alpha/order-renamed equivalent sources yield equivalent frames | pass |
| distinct layouts are not equivalent | pass |
| closed-value (`bool`) facts declare required + forbidden alternatives | pass |
| `reads_query` / `invokes_mutation` relations and mutation effects | pass |
| `reads_state` relation from a `$state` reference | pass |
| unknown component (`Text`, undeclared) fails closed | pass |
| excess positional args (`_args` overflow) fail closed | pass |
| value outside a declared closed domain fails closed | pass |
| unrecognized builtin effect call (`@Run`/`@Set`/`@Count`) fails closed | pass |
| schema-declared default downgrades a matching fact to optional | pass |
| `frame_conflicts` flags a hand-introduced contradiction | pass |
| constraint satisfied by the same candidate, violated by a divergent one | pass |
| constraint check rejects a schema-fingerprint mismatch | pass |
| one-fact counterfactual proves exactly one changed fact | pass |
| counterfactual mutation refuses when no closed-domain fact exists | pass |

The focused new-module suite passed 17 tests
(`tests/test_dsl/test_semantic_frame.py`). A neighboring-directory run
(`test_grammar_capabilities.py`, `test_pack.py`, `test_operator_registry.py`)
passed 44 and reproduced 9 pre-existing failures that are identical with this
change stashed out — this sandbox has no Node.js bridge dependencies
installed (`npm ci` was not run under `src/apps/openui_bridge`), so every
path through the official `@openuidev/lang-core` bridge (`slm_training.dsl`
top-level `validate`, the pack oracle, and the operator registry's
pack-authority validation) raises `RuntimeError: Install bridge deps`. This
module never calls that bridge path directly — it goes through
`slm_training.dsl.parser.validate`, which falls back to the in-process Lark
backend — so none of its own facts depend on the missing bridge. Ruff,
`ruff format`, `compileall`, `scripts.repo_policy`, `scripts.verify_version_stamps
--check`, and `git diff --check` all passed on the touched files.

This is deterministic repository-contract evidence. No corpus synthesis,
train, model eval, benchmark, checkpoint, AgentEvals publication, capability
certificate, or ship claim was produced.

## Honest limitations

- Expression combinators are supported for `BinOp` / `UnaryOp` / `Ternary` /
  `Member` / `Index` / `Arr` / `Obj`, but OpenUI's `@`-prefixed builtin-call
  vocabulary (`@Run`, `@Set`, `@Count`, …) is not yet in the schema's closed
  effect dispatch table and fails closed rather than being guessed a role —
  see the SLM-343 interpretation section above.
- Named/aliased-subterm sharing signaled via the canonical AST's
  `statementId` field is not represented as a frame fact; only structural
  type/prop/order/cardinality/relation/effect facts are captured. This is a
  documented scope limitation, not a silent guess.
- `closed_value_domains` beyond the grammar-intrinsic `bool`/operator-token
  sets are a caller-declared policy, not something re-derived from the
  grammar; a wrong declared domain would produce wrong forbidden facts. The
  shipped `CAP1SchemaV1.from_pack()` factory declares none by default.
- This is a fixture/contract-level module only. No corpus build, train, eval,
  or checkpoint used it.

## Research lineage

The required/optional/forbidden/unspecified fact partition follows the
grounding-contract framing used by semantic-parsing-to-executable-form work
(e.g. abstract-meaning-representation and executable semantic parsing
literature) that separates what an utterance commits to from what remains
open to paraphrase; this implementation is a deterministic, schema-closed
adaptation for one grammar-declared DSL pack, not a learned semantic parser.
