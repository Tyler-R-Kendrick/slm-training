# Language-contract coverage corpus (P2)

**Status:** implemented for OpenUI Lang **0.2.x**.
**Owner module:** [`src/slm_training/data/language_contract/`](../../src/slm_training/data/language_contract/).
**Family:** `language_contract`. **Linear:** SLM-6 (P2).

## Why

The single biggest lever for a specialist DSL is *exhaustive, independently
testable contract coverage* — not prose "semantic descriptions of tokens". This
corpus pins the model to the exact pinned language surface: every grammar
production, lexical form, and component (with its positional props) gets a
minimal **positive**, and every failure mode gets a **negative** that trips a
specific verifier gate.

## What it emits

`build_corpus(split="train")` returns `ExampleRecord`s in two polarities.

### Positives (`polarity="positive"`, `task="generation"`)

- **Productions / lexical forms** — a curated set covering assignment, nested and
  multi-child lists, forward references, integer / negative-number / boolean
  literals, line comments, and enum-valued props.
- **Components** — one minimal, *meaningful* program per component. Instances are
  derived from the authoritative bridge `library_schema` (required props, types,
  enums) rather than a hand-maintained list, so the corpus tracks the component
  library automatically. Every positive:
  - `validate()`s against the pinned bridge,
  - clears `_is_meaningful_program` (has a non-`Stack` component + a placeholder),
  - reaches a non-`Quarantine` tier through the F2 verifier, and
  - is projected via the F1 `emit_record`, so it carries `contract_id`, family,
    lineage, and `split_group_id`.

### Negatives (`polarity="negative"`, `task="adversarial"`)

Each negative is annotated with `meta["expected_gate"]` — the single
[verifier-stack](verifier-stack.md) gate it is designed to trip:

| gate | id | operators |
| --- | --- | --- |
| lexical | G0 | unterminated string · forbidden control char |
| grammar | G1 | missing assignment · unclosed paren · missing list comma |
| schema | G2 | unknown component · too many positional args · missing required prop · literal content prop |
| references | G3 | undefined reference · duplicate binder · missing root · unreachable binder |
| dataflow | G4 | v0.5 `Query` / `Mutation` / `@`-action / `$`-state syntax (outside the pinned 0.2.x contract) |

For G0–G3 the targeted gate is also the *first* failing gate
(`verify_record(...).failing_gate`). v0.5 dataflow syntax (G4) is invalid at
several levels at once, so those are asserted only through the isolated
`evaluate_gate(Gate.DATAFLOW, ...)` check.

## Coverage report

`coverage_report()` returns covered vs. total for components, required prop
positions, productions, and gates. Current tree: **54/54 components**, **88/88
required prop positions**, all five negative gates, 9 production positives. It is
computed from the static `openui_prop_order.json` inventory so it needs no bridge.

## Scope notes

- Optional prop positions beyond the required set are not exhaustively enumerated;
  the report tracks required-position coverage (the schema-mandated surface).
- Wiring this family into the reproducible build belongs to **P12** (the sole
  writer of `harnesses/train_data/pipeline.py`), which will add a
  `_records_from_language_contract` loader calling `build_corpus`.
