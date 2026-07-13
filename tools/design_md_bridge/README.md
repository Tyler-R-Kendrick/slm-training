# DESIGN.md bridge (`@google/design.md`)

Node sidecar that wraps the official Google DESIGN.md linter for Python harnesses.

## Setup

```bash
cd tools/design_md_bridge
npm ci
```

## CLI

```bash
echo '{"op":"lint","source":"---\\nname: Demo\\ncolors:\\n  primary: \\"#000000\\"\\n---\\n\\n## Overview\\nDemo.\\n"}' \
  | node cli.mjs
```

Returns `{ ok, score, summary, findings }`.
