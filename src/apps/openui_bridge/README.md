# OpenUI Lang bridge (`@openuidev/lang-core` + official `openuiLibrary`)

Node sidecar that wraps the official OpenUI parser / serializer / prompt generator for the Python harnesses.

## Setup

```bash
cd src/apps/openui_bridge
npm ci   # or: npm install
```

## CLI

```bash
echo '{"op":"validate","source":"root = Stack([cta])\ncta = Button(\":cta.label\")"}' \
  | node cli.mjs
```

Ops: `parse`, `validate`, `serialize`, `prompt`, `schema`, `stream_check`.

The pinned `@openuidev/lang-core@0.2.9` bridge exposes prototype v0.5 parsing
capabilities. Parsing and streaming retain state declarations plus `Query`,
`Mutation`, `Action`, and tool-call metadata. Whole-program serialization
preserves validated source when runtime sidecars would otherwise be lost. The
model's honest layout-only training contract remains the published 0.2.x subset.

## Library

[`library.mjs`](library.mjs) re-exports official `openuiLibrary` from `@openuidev/react-ui/genui-lib` (~54 components, root `Stack`).

User-facing string props (`text`, `label`, `title`, `placeholder`, `alt`, …) must be placeholder tokens (`:hero.title`). Layout enums use official values (`column`/`row`, gap `none`…`2xl`).

## Language contract and fixture migration

Python owns the canonical 16-hex identity in
[`language_contract.py`](../../slm_training/dsl/language_contract.py); the
bridge does not mint a second contract family. Grammar and tokenizer changes
therefore produce a new dataset contract automatically. DSL-tokenizer v1
artifacts are not compatible with the v2 state/builtin alphabet and must be
rebuilt.
