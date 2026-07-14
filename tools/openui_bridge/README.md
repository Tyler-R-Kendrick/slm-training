# OpenUI Lang bridge (`@openuidev/lang-core` + official `openuiLibrary`)

Node sidecar that wraps the official OpenUI parser / serializer / prompt generator for the Python harnesses.

## Setup

```bash
cd tools/openui_bridge
npm ci   # or: npm install
```

## CLI

```bash
echo '{"op":"validate","source":"root = Stack([cta])\ncta = Button(\":cta.label\")"}' \
  | node cli.mjs
```

Ops: `parse`, `validate`, `serialize`, `prompt`, `schema`, `stream_check`.

The bridge implements the OpenUI Lang **v0.5 language specification** using
`@openuidev/lang-core@0.2.9`. The package version and language version are
separate: `0.2.9` is the installed implementation of the v0.5 syntax. Parsing
and streaming retain state declarations plus `Query`, `Mutation`, `Action`, and
tool-call metadata. Whole-program serialization preserves the validated source
so these non-render-tree statements and expression precedence are not lost.

## Library

[`library.mjs`](library.mjs) re-exports official `openuiLibrary` from `@openuidev/react-ui/genui-lib` (~54 components, root `Stack`).

User-facing string props (`text`, `label`, `title`, `placeholder`, `alt`, …) must be placeholder tokens (`:hero.title`). Layout enums use official values (`column`/`row`, gap `none`…`2xl`).

## Language contract and fixture migration

[`../../grammars/openui_contract.json`](../../grammars/openui_contract.json)
pins the v0.5 spec, the installed parser source and package integrity, component
schema, canonicalizer, renderer, and DSL-tokenizer version. The runtime adds a
canonical tool-schema hash and emits the resulting `contract_id` from every
bridge operation. Python `ExampleRecord` objects carry the same identifier.

Legacy JSONL fixtures do not need an eager rewrite: loading a record without
`contract_id` deterministically stamps the current contract (including any
`meta.tool_schema`), and the next serialization persists it. Tokenized artifacts
from DSL tokenizer v1 are not contract-compatible with v2 and must be rebuilt.
