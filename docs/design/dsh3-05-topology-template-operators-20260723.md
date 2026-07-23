# DSH3-05 topology and template-oriented operators

SLM-373 extends the pack-owned SLM-372 library with exact topology rewrites and
explicit template aliases. The composed OpenUI library retains all six core
local operators and adds:

| Operator | Exact behavior | Cost | Inverse family |
| --- | --- | ---: | --- |
| `openui.move_node` | detach one inline subtree and insert it into a schema-compatible child role | 1.5 | itself with original parent/index |
| `openui.reparent_node` | exact alias of `move_node` | 1.5 | itself |
| `openui.wrap_node` | replace a node with an explicit empty wrapper template containing that node | 1.5 | `unwrap_node` |
| `openui.unwrap_node` | replace an exact single-child wrapper with its element child | 1.0 | `wrap_node` with a wrapper alias |
| `openui.duplicate_subtree` | deep-copy an inline subtree into a compatible child role | 2.0 | — |
| `openui.expand_template` | replace an exact contracted subtree with its manifest expansion | 2.0 | `contract_subtree` |
| `openui.contract_subtree` | replace an exact expansion with its manifest contraction | 2.0 | `expand_template` |

Every successful mutation emits through
`openui.production_codec.statement_bindings` and returns to the ordinary
OpenUI pack authority path. No topology operator splices source text or
bypasses parse/schema, scope, property-order, canonicalization, or canonical
round-trip checks.

## Topology boundary

Move, reparent, and duplicate accept only inline element subtrees. Top-level
binding deletion/reparenting is unsupported because the binding reference
graph would also need an exact rewrite proof. A destination role is owned by
the selected parent descriptor, must be an array role, and must admit the
subtree component under the component schema.

Ordered insertion refs remain bound to the destination node, child role, exact
parent-order digest, and position. Same-role moves adjust the before-state
boundary after detachment and reject semantic no-ops. Moving a node into
itself or a descendant returns `topology.cycle`.

Subtrees containing statement-binding refs return
`topology.capture_unsupported`. This is the current conservative binder policy:
no operator invents alpha-renaming or capture avoidance without a compiler
proof. Ordinary pack scope extraction still runs after every capture-free
rewrite.

## Explicit template aliases

`OpenUITemplateAliasV1` is an immutable manifest containing:

- owning pack ID;
- exact expanded and contracted element ASTs;
- source artifact digest;
- fixed canonical lowering ID;
- optional wrapper child role.

The model-visible `TemplateRef` still exposes only its DSH3-03 opaque
descriptor. Compiler-private context binds it to a manifest. Template
applications record the template fingerprint, source artifact digest, pack,
lowering ID, operation, and before/after subtree digests directly in the
`ActionEffectV1` topology delta.

Wrap requires an empty declared child role and schema-compatible wrapped
component. Expand/contract require byte-for-byte canonical binding-AST
identity with the manifest side being replaced. Missing aliases omit the
template-dependent operators from the pack library, so lookup/application
returns stable `operator.unsupported`; mismatches and invalid manifests use
`template.*` rejection codes.

## Stable failures

The implementation uses stable codes including:

- `topology.root_move`
- `topology.cycle`
- `topology.capture_unsupported`
- `topology.incompatible_cardinality`
- `topology.incompatible_child`
- `topology.index_role_mismatch` / `topology.invalid_index`
- `topology.no_change`
- `topology.unsupported`
- `template.mismatch`
- `template.unsupported`
- `template.provenance_invalid`
- `template.duplicate`

DSH3-03 `ref.*`, SLM-372 `local.*`, and registry `operator.unsupported` codes
remain intact.

## Evidence and scope

Deterministic fixtures cover both move aliases and inverse restoration,
cross-parent duplication, ordered insertion, cycle rejection, capture
rejection, wrapper inverse restoration, invalid unwrap cardinality,
expand/contract inverse restoration, ordinary production-codec construction
equivalence, explicit template evidence, immutable alias input, invalid pack
provenance, missing capability, declaration costs/locality/inverses, and
composition with the six core local operators.

These are compiler contract/unit fixtures. This change does not enumerate the
complete legal action set, synthesize operator training data, train or evaluate
a model, change ship gates, create a checkpoint, or claim topology-diffusion
quality.
