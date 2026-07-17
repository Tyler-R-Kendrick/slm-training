# G3 (SLM-47): the latent-DSL generator — task → grammar → instantiated pack

**Status:** landed (mechanical pack instantiation; no LLM, no training run).
**Code:** `src/slm_training/dsl/latent/__init__.py` (spec + synthesizer),
`scripts/synthesize_pack.py` (entrypoint),
`tests/test_dsl/test_latent_pack.py` (round-trip proof),
`tests/test_dsl/fixtures/latent_task.json` (sample spec).
**Depends on:** F1 (`docs/design/dsl-pack-contract.md`) making pack creation
mechanical. **Feeds:** G4 (a latent DSL per task as the reasoning substrate).

## What G3 is

Grammar Prompting (Wang et al., NeurIPS 2023,
[arXiv:2305.19234](https://arxiv.org/abs/2305.19234)) prompts a **frozen large
model** with a task-specific grammar. G3 instead **instantiates a full DSL pack
automatically** — grammar → backend → scope rules → placeholder policy → prop
order → incremental engine — so the DSL itself, not a prompt to a frozen model,
becomes the reasoning substrate. The honest G3 shipped here proves that
instantiation is **mechanical**: given a typed task inventory, a complete
(partial-by-contract) `DslPack` is produced deterministically and registered.

The "an LLM synthesizes the grammar" step is **explicitly stubbed** with a
deterministic emitter. That is the honest version of the Grammar-Prompting move:
the shape of the pipeline (task → grammar → codec → oracle → corpus → tiny
model) is real; the frontier-model and trained-model rungs are deferred, not
faked.

## The mechanism

1. **`LatentTaskSpec`** (`task_id`, `description`, ordered `components`, each a
   `LatentComponent(name, props)`; `root_name`). `props` doubles as the
   positional argument names (its length is the component's arity) and as the
   pack's `prop_order` entry. `from_dict` parses inline/fixture JSON;
   `dsl_id` is the deterministic slug `latent-<task_id>`.
2. **`synthesize_grammar(spec) -> (lark_text, prop_order)`** — deterministically
   emits a minimal `.lark` conforming to the `toy_layout` skeleton
   (`start`/`statement`/`call`/`list`/`STRING`/`NAME`) with a task-derived
   header comment, plus a `prop_order` dict from the spec. No LLM.
3. **`synthesize_pack(spec, *, grammars_dir=None) -> DslPack`** — writes the
   `.lark`, builds + `register_backend`s a `LarkFileBackend` over it, then
   assembles + `register_pack`s a PARTIAL `DslPack` and returns it. Idempotent
   and re-registration-safe (rewrites the file, overwrites registry entries).
   It first forces the builtin backend/pack registries to load (`list_packs()`,
   `available_backends()`) so the new entry does not short-circuit their lazy
   "empty registry" guards.

Because component names ride the generic `NAME` call rule (they are *not*
promoted to per-name terminals), the generic Lark transformer yields ElementNode
ASTs and the grammar-generic `extract_scope_slices` works with zero extra
wiring — the same free-transformer path `toy-layout` uses.

### Grammar-file lifetime

`LarkFileBackend` reads the grammar lazily (on first `parse`), so the file must
outlive pack use. The default location `<grammars>/latent/<dsl_id>.lark`
persists; passing a temp `grammars_dir` (e.g. pytest `tmp_path`, or the
entrypoint's `outputs/dsl/latent/`) gives an ephemeral lifetime.

## Filled vs honest-None vs deferred

| Slot | State | Why |
| --- | --- | --- |
| `backend` | **FILLED** | `LarkFileBackend` over the synthesized `.lark` |
| `scope_extractor` | **FILLED** | grammar-generic `extract_scope_slices(dsl=dsl_id)` |
| `placeholder_policy` | **FILLED** | shared `PLACEHOLDER_RE` / `CONTENT_PROPS` / `canonical_slot_contract` |
| `prop_order` | **FILLED** | from the task spec's component props |
| `incremental_engine` | **FILLED** | `OpenUIIncrementalEngine(grammar_path)` fastpath |
| `canonicalize` | **honest-None** | production codec is still OpenUI-shaped (uppercase-call); blocked on F2 |
| `oracle` | **honest-None** | same — no behavioral verdict without a codec/generator |
| `corpus_generator` | **honest-None** | needs a synthesized component JSON schema (deferred) |

`reward_label = "parse_only"` (same honesty as `toy-layout`).

### Explicitly deferred (documented, not faked)

- **The trained tiny model per task.** The issue's end-goal "grammar → … →
  tiny trained model" — G3 stops at pack instantiation; the per-task model is
  G4's job on top of this substrate.
- **A synthesized component JSON schema for `ProgramGenerator`.** Needed for the
  `oracle` + `corpus_generator` slots (generation/verification), not for the
  round-trip. Deferred.
- **A synthesized canonicalizer / oracle.** Blocked on F2's codec
  generalization — the production codec's lexical layer still expects uppercase
  component heads (`docs/design/dsl-pack-contract.md` ~99-104), which is exactly
  why `canonicalize`/`oracle` are honest-`None` here rather than pretend
  implementations.
- **The real LLM task→grammar step.** Stubbed with the deterministic emitter.
- **Per-name terminal restriction of `call`.** Would make the grammar reject
  undeclared component names, but drops the free generic-transformer fastpath.

## Round-trip evidence (executed, not hypothetical)

`tests/test_dsl/test_latent_pack.py` synthesizes a pack from a trivial 3-component
spec into a `tmp_path` grammars dir and asserts:

1. **backend round-trip** — `parse(PROGRAM)` → `serialize` → re-`parse`, ASTs equal;
2. **scope rules** — `scope_extractor` yields exactly `{document, statement,
   expression, lexical}`;
3. **prop order** — `prop_order()["field"] == ["label", "value"]` (from the spec);
4. **placeholder policy** — `slot_contract` == the extracted placeholders;
5. **incremental engine** — `can_complete_with_holes("root = row(")` is true and
   `set_prefix(...).next_terminals()` is non-empty at the frontier;
6. **honest-None** — `require("oracle" | "canonicalize" | "corpus_generator")`
   raises `PackSlotUnavailable`;
7. **clean globals** — pack + backend are popped in `try/finally`; `list_packs()`
   and `available_backends()` are asserted unchanged before/after;
8. **determinism + idempotence** — `synthesize_grammar` is byte-stable and a
   second `synthesize_pack` re-registers safely.

Entrypoint demo (`python -m scripts.synthesize_pack --spec
tests/test_dsl/fixtures/latent_task.json`):

```
task 'kv-form' -> pack 'latent-kv-form'
grammar: outputs/dsl/latent/latent-kv-form.lark
filled slots: ['pack_id', 'backend', 'placeholder_policy', 'reward_label', 'scope_extractor', 'prop_order', 'incremental_engine']
reward_label: parse_only
prop_order: {'row': ['children', 'gap'], 'field': ['label', 'value'], 'text': ['text', 'size']}
demo program:  'root = row(":demo.children", ":demo.gap")'
round-trip root-stable: True
scopes: ['document', 'expression', 'lexical', 'statement']
slot_contract: (':demo.children', ':demo.gap')
incremental engine can_complete_with_holes('root = row('): True
```

## Run metadata

| Field | Value |
| --- | --- |
| Device | CPU |
| Steps | n/a (no training) |
| Backend | synthesized-lark (`LarkFileBackend` over generated `.lark`) |
| n | fixture count = 6 tests in `test_latent_pack.py` (1 synthesized spec + inline variants) |
| Honesty mode | mechanical-instantiation-only |
| Ship-gate | n/a (no checkpoint, no eval suite) |
| Checkpoint | none — `docs/MODEL_CARD.md` deliberately **not** updated |
| Date | 2026-07-17 |

## Checks

- `tests/test_dsl` — 161 passed, 1 skipped (was 19 pack tests; +6 latent tests,
  no pre-existing outcome changed; `.githooks/check-changed` also ran
  `tests/test_scripts` + `tests/test_harnesses/model_build`: 330 passed total).
- `scripts/repo_policy` — ok; ruff — clean on all three new files.

## Cross-links

- F1 pack contract: `docs/design/dsl-pack-contract.md` (names G3 at ~line 8).
- Research lineage Track G rows: `docs/design/research-lineage.md` (~531).
