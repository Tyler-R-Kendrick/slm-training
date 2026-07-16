# E200–E204 — generated layout-role constraints

Status: **diagnostic only; no checkpoint promoted; ship gates not run**.

## Hypothesis

E199 assigned a primitive string to a referenced child binder. The fix must
follow roles already represented by the Lark grammar and generated schema, not
match a known prompt, component, or output. A canonical corpus audit parsed all
496 E177 records and found 1,916 declarations: every declaration RHS has a
component AST surface and none has a primitive surface. The decoder therefore
constrains any typed declaration value to grammar component candidates.

The same generated-schema mechanism identifies `children` arrays and canonical
content properties. Array elements are restricted to grammar node terminals;
content properties use the supplied slot-contract symbols exactly. These are
role rules from the grammar, schema, and centralized `CONTENT_PROPS` contract.

## Matched train

E201 used the same committed corpus, balanced-mixture hash, CPU, 32 steps,
batch 4, seed 0, frozen SmolLM2-135M context, schema/slot context, no DESIGN.md
context, alignment weight 1.0, and no checkpoint sync as E196. Component
alignment was split by typed binder identity into root and bound roles.

| Last loss | Wall s | Aligned rows | Root component | Bound component | Binder | Structural | Symbol | Literal |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9.1521 | 112.04 | 757 | 128 | 126 | 128 | 128 | 128 | 119 |

Alignment loss fell from 69.5053 to 3.8338 (mean 13.5849). The checkpoint is
local scratch evidence only; its SHA-256 and complete recipe are in
[the result JSON](iter-e200-e204-layout-role-compiler-20260716.json).

## Evaluations

All rows are strict one-example smoke diagnostics with no unconstrained
fallback. Each emitted AgentEvals JSONL and an AgentV SDK bundle.

| Experiment | Generalized change | Syntax | Meaningful parse | Structure | Component recall | Placeholder validity | p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E200 | All declaration RHS values are component-role | 1.0 | 0.0 | 0.1917 | 0.0 | 0.0 | 5686.16 |
| E202 | Matched root/bound role-aligned E201 checkpoint | 0.0 | 0.0 | 0.0467 | 0.0 | 0.0 | 17477.16 |
| E203 | Generated `children` property accepts node terminals | 0.0 | 0.0 | 0.0955 | 0.25 | 0.55 | 12224.90 |
| E204 | Canonical content properties accept contract symbols only | 0.0 | 0.0 | 0.0955 | 0.25 | 0.70 | 12650.16 |

E200 proves the declaration-role rule closes the primitive-RHS hole, but the
older checkpoint chooses the wrong bound component. E201 then learns the split
roles but E202 recursively emits legal expressions. E203 removes arbitrary
literals from generated children collections and recovers component and
placeholder signals. E204 removes fixed literals from content slots and raises
placeholder validity to 0.70. It still emits 102 tokens and stops at the token
cap inside nested `children`; syntax, meaningful parse, and reward remain zero.
All rows report zero compiler fallback.

This is a negative ship result. Improved submetrics do not make the checkpoint
promotable.

## Next hypothesis

Measure close-versus-extend decisions and gold AST cardinality across the
committed corpus, then derive collection-completion pressure from grammar/schema
structure and learned cardinality. Do not add stopping cases for observed
component arrangements or prediction strings.
