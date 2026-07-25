# DSH3-04 core local OpenUI operators

SLM-372 implements the first executable OpenUI operator family on the DSH3-01
contracts, DSH3-02 pack authority boundary, and DSH3-03 opaque references. The
inventory is deliberately limited to six exact statement-structural edits:

| Operator | Typed arguments | Declared effect | Inverse where defined |
| --- | --- | --- | --- |
| `openui.add_child` | parent, role, template, optional state-bound index | cardinality + topology | `remove_node` |
| `openui.remove_node` | non-root inline node | cardinality + topology | `add_child` |
| `openui.replace_node` | node, compatible template | topology | — |
| `openui.set_property` | node, owning role, typed value | property | `unset_property` |
| `openui.unset_property` | node, owning optional role | property | `set_property` |
| `openui.reorder_children` | parent, owning array role, exact permutation | topology | itself |

Every executor is pure over `OperatorStateV1`. It parses the canonical
statement-binding AST, deep-copies compiler-supplied templates/values, emits
through the canonical production codec, and then returns to the ordinary
pack-owned apply path. A successful application therefore still requires
parse/serialize, static/schema oracle, scope extraction, property order,
canonicalization, and canonical round-trip checks.

## Compiler context and legality

`build_openui_local_operator_context` derives node, property-role, and
ordered-parent index descriptors from one validated state. Serialized
candidate surfaces contain only DSH3-03 descriptors and opaque refs. Compiler
private payloads hold AST paths, property names, and template/value objects;
those payloads are neither descriptor fields nor model-visible identifiers.

Roles come from the owning component schema. Child insertion and replacement
check the schema's allowed component references before emission. Required
properties cannot be removed. Index refs bind the exact node, child role,
parent-order digest, and insertion position. Reordering requires a complete
permutation with no duplicates or omissions.

OpenUI's current statement codec serializes component properties positionally.
An edit that would require a missing optional predecessor, or removal that
would leave a populated optional successor, is rejected as
`local.unsupported_pack_semantics`. This prevents a nominal one-property edit
from silently adding a `null` sibling property. Canonical emission that erases
the requested edit is rejected by the same code.

## Stable rejection boundary

The registry now preserves an executor's `OperatorRejectedError` code and
failed precondition instead of collapsing all exact precondition failures into
`operator.arguments_rejected`. The local family uses stable codes including:

- `local.root_deletion`
- `local.required_property_removal`
- `local.incompatible_replacement`
- `local.invalid_index` / `local.invalid_order`
- `local.role_mismatch` / `local.index_role_mismatch`
- `local.property_missing` / `local.property_value_invalid`
- `local.child_role_required` / `local.unsupported_role`
- `local.no_change`
- `local.unsupported_pack_semantics`

DSH3-03 `ref.*` failures propagate unchanged. Unsupported top-level binding
removal, non-element templates, v0.5 typed programs, and pack/schema semantics
that cannot be expressed exactly fail closed; they do not fall back to text
replacement.

## Evidence and scope

Deterministic tests cover all six declarations; pure dry-run/apply identity;
pack-valid success; root/required/incompatible/order/role/stale failures;
opaque-ref permutation; property and child inverse restoration; minimal AST
diff locality; every insertion position for zero-, one-, and two-child states;
and every non-identity permutation of three children. These are contract and
unit fixtures, not train/eval evidence.

This change does not enumerate the complete legal action set, add training
data, train a model, alter ship gates, create a checkpoint, or claim CAP2
capability.
