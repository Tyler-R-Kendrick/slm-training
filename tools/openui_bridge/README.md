# OpenUI Lang bridge (`@openuidev/lang-core`)

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

Ops: `parse`, `validate`, `serialize`, `prompt`, `schema`.

## Library

[`library.mjs`](library.mjs) defines the training subset with official `defineComponent` + `createLibrary`: `Stack`, `Card`, `Text`, `Button`. Content props must be placeholder strings (`:hero.title`).
