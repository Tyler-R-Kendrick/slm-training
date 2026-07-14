# OpenUI dataset contract identity (`contract_id`)

**Status:** implemented for OpenUI Lang **0.2.x** (layout subset).
**Owner module:** [`src/slm_training/dsl/contract.py`](../../src/slm_training/dsl/contract.py).
**Linear:** SLM-1 (F0).

## Why

Every training / eval record is produced against a specific *language contract* —
the language spec, the parser, the component and tool schemas, the canonicalizer,
the renderer, and the DSL tokenizer. If **any** of these change, records built
before and after are no longer comparable: a component-library bump silently
changes the target distribution, and a train/eval split assembled across two
contracts leaks or drifts without anyone noticing.

`contract_id` makes that dependency explicit and reproducible. It is a stable
hash of the seven contract inputs, stamped into every record's
`meta["contract_id"]` and into each build's `manifest.json` / `stats.json`.
**A change to any input is, by definition, a new dataset version.**

```
contract_id = hash(lang_spec_version + parser_commit + component_schema
                   + tool_schema + canonicalizer + renderer
                   + tokenizer_version)
```

Concretely: `contract_components()` returns the seven strings below,
`contract_id()` returns `"oc-" + sha256(canonical_json(components))[:16]`.

## The seven inputs

| input | how it is derived | current value (0.2.x) |
| --- | --- | --- |
| `lang_spec_version` | `@openuidev/lang-core` version in `tools/openui_bridge/package.json` | `0.2.9` |
| `parser_commit` | lang-core version + sha256 of `grammars/openui.lark` | `lang-core@0.2.9+lark:<hash>` |
| `component_schema` | `@openuidev/react-ui` version + sha256 of `grammars/openui_prop_order.json` | `react-ui@0.12.1+prop_order:<hash>` |
| `tool_schema` | tool constructs in the contract (none in 0.2.x; bridge sets `toolCalls=false`) | `none@0.2.x` |
| `canonicalizer` | `strip_style_literals` + lang-core serialize + production codec, versioned by `CANONICALIZER_VERSION` | `strip_style_literals+lang-core-serialize@v1` |
| `renderer` | `@openuidev/react-ui` + `@openuidev/react-lang` versions | `react-ui@0.12.1+react-lang@0.2.8` |
| `tokenizer_version` | `DSL_TOKENIZER_VERSION` read from `models/dsl_tokenizer.py` | `dsl-tok@v1` |

### Design properties

- **Deterministic** — the same tree always yields the same id. Inputs are static
  repo artifacts (the bridge `package.json`, the vendored grammars, in-source
  version constants), hashed with `sort_keys=True` canonical JSON.
- **No heavy deps** — computing `contract_id` requires **neither** the Node
  bridge **nor** torch. The tokenizer version is read from source with a regex
  rather than importing `models.dsl_tokenizer` (which would pull torch via
  `models/__init__`). This keeps the id computable in any pure-data build.
- **Human-readable components** — each input is a legible string, so a diff of two
  `manifest.json` files shows *what* changed, not just that the hash moved.

## Where it is stamped

- **Records:** `harnesses/train_data/pipeline.py::_normalize_record` and
  `harnesses/test_data/pipeline.py::_normalize` add `meta["contract_id"]` to
  every emitted `ExampleRecord` (no new wire field — it lives in `meta`, per the
  F1 schema conventions).
- **Builds:** both build pipelines write `contract_id` into `stats.json` and a
  full `contract` fingerprint (`{contract_id, components}`) into `manifest.json`.

Downstream code reads `record.meta["contract_id"]` (or `dsl.contract_id()`) to
assert that a split / eval / mixture never mixes contracts.

## When to bump

Bump `CANONICALIZER_VERSION` in `contract.py` when canonicalization changes the
bytes emitted for a program. Everything else moves on its own: upgrading a
`@openuidev/*` package, editing `grammars/openui.lark` or
`grammars/openui_prop_order.json`, or bumping `DSL_TOKENIZER_VERSION` all change
`contract_id` automatically. Treat any change as a new dataset version — rebuild
train and test data together so their `contract_id`s match.

## v0.5 migration (pending upstream package)

SLM-1's original scope was to move the contract to full **OpenUI Lang v0.5**
(state / query / mutation / action / tool constructs). The mandatory "Verify
FIRST" pre-flight found that **no v0.5 package or spec exists yet**:
`@openuidev/lang-core` is published only through `0.2.9`, `@openuidev/react-lang`
through `0.2.8`, and there is no v0.5 spec in the repo. The current grammar and
bridge are layout-only (`tools/openui_bridge/cli.mjs` hardcodes
`toolCalls=false`).

Per the issue's escalation clause this was raised to the repo owner, and F0 was
**scoped to the `contract_id` reproducibility half** against the 0.2.x contract.
`tool_schema` is therefore `none@0.2.x`, and the acceptance criterion "bridge
round-trips a v0.5 program containing state + query + mutation + action + tool
call" is deferred until a v0.5 package / spec is available.

When v0.5 lands, the upgrade is mechanical and `contract_id` already captures it:

1. Bump the `@openuidev/*` deps in `tools/openui_bridge/package.json`
   (`lang_spec_version`, `component_schema`, `renderer` all move).
2. Extend `grammars/openui.lark` + `grammars/openui_prop_order.json` for
   state/query/mutation/action/tool productions (`parser_commit`,
   `component_schema` move).
3. Replace `tool_schema` `none@0.2.x` with the real v0.5 tool-schema hash and
   flip the bridge's `toolCalls` handling.
4. Extend `dsl/production_codec.py`, `models/dsl_tokenizer.py`
   (`tokenizer_version` moves), and `dsl/openui_tokens.py` for the new
   constructs.

Each of those changes shifts `contract_id`, so any v0.5 corpus is automatically a
distinct dataset version from the 0.2.x corpus — no accidental mixing.
