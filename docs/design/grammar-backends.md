# Grammar-based training (pluggable DSLs)

Training and constrained decode are **DSL-agnostic**. OpenUI is the default
frontend grammar; the same stack can train against any backend that implements
`GrammarBackend`.

## Backends

| id | kind | AST source |
|---|---|---|
| `openui` (default) | hybrid | Official `@openuidev/lang-core` when the Node bridge is installed; otherwise Lark |
| `openui-langcore` | lang-core | Official ElementNode via `tools/openui_bridge` |
| `openui-lark` | lark | In-process Lark parse of `grammars/openui.lark` → ElementNode-like dict |
| `toy-layout` | lark | Example alternate DSL (`grammars/toy_layout.lark`) |

Register more DSLs with `register_backend(...)` or drop a `.lark` file and wrap
it with `LarkFileBackend`.

## Real AST extraction

- **Official path:** `get_backend("openui-langcore").parse(src).root` is the
  ElementNode dict from `createParser` / `jsonToOpenUI`.
- **Lark path:** `grammars/openui.lark` + `grammars/openui_prop_order.json`
  (schema prop order snapshot) map positional calls onto named props
  (`children`, `text`, `direction`, …) so the tree shape matches lang-core for
  common components.
- Fingerprints: `component_multiset` / `ast_fingerprint` in
  `slm_training.dsl.grammar.backends.ast_utils`.

## Training / eval switch

```bash
train-model --grammar-dsl openui-lark ...
evaluate-model --grammar-dsl toy-layout ...
# or
export SLM_GRAMMAR_DSL=openui-lark
```

`ModelBuildConfig.grammar_dsl` activates the backend for stream checks,
structural bias, and (via `dsl.parser`) parse/validate when that module is used.

## Adding a new DSL

1. Author `grammars/<name>.lark` (assignment / call / list shape works with the
   generic transformer).
2. Optionally ship a prop-order JSON for named props.
3. Subclass `LarkFileBackend` or implement `GrammarBackend` in
   `grammar_backends/`.
4. `register_backend(YourBackend())` inside `_ensure_builtins` (or at import).
5. Train with `--grammar-dsl <id>`.

Official schema validation and placeholder content policy remain lang-core
responsibilities when that backend is selected; Lark backends enforce syntax +
root presence and leave library policy to the caller.
