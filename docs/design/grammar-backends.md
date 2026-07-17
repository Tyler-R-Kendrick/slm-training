# Grammar-based training (pluggable DSLs)

Training and constrained decode are **DSL-agnostic**. OpenUI is the default
frontend grammar; the same stack can train against any backend that implements
`GrammarBackend`.

## Backends

| id | kind | AST source |
|---|---|---|
| `openui` (default) | hybrid | Official `@openuidev/lang-core` when the Node bridge is installed; otherwise Lark |
| `openui-langcore` | lang-core | Official ElementNode via `src/apps/openui_bridge` |
| `openui-lark` | lark | In-process Lark parse of `src/slm_training/dsl/grammars/openui.lark` → ElementNode-like dict |
| `toy-layout` | lark | Example alternate DSL (`src/slm_training/dsl/grammars/toy_layout.lark`) |

Register more DSLs with `register_backend(...)` or drop a `.lark` file and wrap
it with `LarkFileBackend`.

## Real AST extraction

- **Official path:** `get_backend("openui-langcore").parse(src).root` is the
  ElementNode dict from `createParser` / `jsonToOpenUI`.
- **Lark path:** `src/slm_training/dsl/grammars/openui.lark` + `src/slm_training/dsl/grammars/openui_prop_order.json`
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

1. Author `src/slm_training/dsl/grammars/<name>.lark` (assignment / call / list shape works with the
   generic transformer).
2. Optionally ship a prop-order JSON for named props.
3. Subclass `LarkFileBackend` or implement `GrammarBackend` in
   `grammar_backends/`.
4. `register_backend(YourBackend())` inside `_ensure_builtins` (or at import).
5. Train with `--grammar-dsl <id>`.

Official schema validation and placeholder content policy remain lang-core
responsibilities when that backend is selected; Lark backends enforce syntax +
root presence and leave library policy to the caller.

## DSL packs (F1, SLM-34)

The backend covers only the parse/decode half. The full per-DSL bundle —
{grammar backend, canonicalizer, validity oracle, typed corpus generator,
scope check, placeholder policy} — is a **pack**:
[`src/slm_training/dsl/packs/`](../../src/slm_training/dsl/packs/)
(`DSLPack` in `types.py`; registry `get_pack`/`register_pack`/`list_packs`
mirroring this backend registry). Builtin instances: `openui` (codec
round-trip canonicalizer, `validate_output` oracle, progspec generator) and
`toy-layout` (identity canonicalizer + template generator — the
contract-shape guard). `grammar` and `validity_oracle` are independent pack
fields on purpose: the F4 ontology packs validate by consistency check, not
CFG parse.

Pack seams added in F1: `fastpath/engine.engine_for_dsl` resolves any
registered backend's `grammar_path` (no more hard-coded path table), and
`data/verify/stack._grammar_backend` accepts a backend id (default remains
the OpenUI Lark backend — the G0–G12 gate stack itself is OpenUI-semantic
and stays that way until a pack supplies its own semantic gates).

Adding a DSL end-to-end is now: backend (steps above) → `DSLPack` bundle →
`register_pack` → `tests/test_dsl/test_packs.py` contract invariants pass.
