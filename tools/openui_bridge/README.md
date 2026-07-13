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

## Library

[`library.mjs`](library.mjs) re-exports official `openuiLibrary` from `@openuidev/react-ui/genui-lib` (~54 components, root `Stack`).

User-facing string props (`text`, `label`, `title`, `placeholder`, `alt`, …) must be placeholder tokens (`:hero.title`). Layout enums use official values (`column`/`row`, gap `none`…`2xl`).
