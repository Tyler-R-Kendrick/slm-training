# F1 (SLM-34): the DSL-pack contract — OpenUI as the first pack

**Status:** landed (refactor + contract formalization; no training behavior change).
**Code:** `src/slm_training/dsl/pack.py` (contract + registry),
`tests/test_dsl/test_pack.py` (registry, slots, e2e fixture run).
**Blocks unblocked:** F2 (GraphQL), F3 (patterns DSL,
[design-patterns-dsl.md](design-patterns-dsl.md)), F4 (nomenclatures,
[expert-nomenclature-packs.md](expert-nomenclature-packs.md)), G3 (latent-DSL
generator, **landed** — [latent-dsl-generator.md](latent-dsl-generator.md):
`synthesize_pack(task_spec)` mechanically instantiates a partial pack from a
synthesized grammar, the same filled/honest-None split as `toy-layout`).

## The contract

A **DSL pack** is one frozen `DslPack` value bundling everything the training
stack needs to treat a language as a first-class target. The slot list is the
pack-contract table already committed in
[design-patterns-dsl.md](design-patterns-dsl.md) ("Pack contract mapping
(F1)"), field-for-field:

| Pack slot (F3 table) | `DslPack` field | OpenUI provider |
| --- | --- | --- |
| grammar | `backend: GrammarBackend` | `dsl/grammar/backends` registry (`get_backend("openui")`, hybrid lang-core/Lark) |
| validity oracle | `oracle` | G0–G12 gate stack, `data/verify/stack.py` (`verify_record`) |
| typed-AST generator | `corpus_generator` | `data/progspec/generate.py` `ProgramGenerator` factory |
| canonicalizer | `canonicalize` | D2 confluent codec round-trip, `dsl/canonicalize.py` |
| scope rules | `scope_extractor` | grammar-generic `data/scope_extract.extract_scope_slices` |
| placeholder policy | `placeholder_policy: PlaceholderPolicy` | `dsl/placeholders.py` regex + content props + `canonical_slot_contract` |

Plus contract metadata the F3/F4 docs require:

- `reward_label: str` — **oracle honesty label**. OpenUI's is
  `"well_formed_not_behavioral"`: the gate stack proves grammar/schema/
  reference/canonical well-formedness, not behavior (runtime/behavior gates
  only fire when evidence is supplied). Any reward derived from a pack's
  oracle must carry this label. toy-layout's is `"parse_only"`.
- `prop_order` and `incremental_engine` — optional operational slots
  (positional-prop declaration order for the production codec; streaming
  DFA engine for constrained decode, consulted by
  `dsl/grammar/fastpath/engine.engine_for_dsl` before its alias list).
- Slots are typed as Protocol-shaped callables/objects
  (`Canonicalizer`, `ValidityOracle`), **not** concrete Lark paths — so the
  F4 ontology variant (grammar → graph-walk constraint, oracle → ontology
  reasoner) fills the same slots without changing the contract.

**Partial packs fail closed.** Slots a language genuinely does not provide
are `None`; `DslPack.require(slot)` raises `PackSlotUnavailable` naming the
pack, the missing slot, and the filled slots. `toy-layout` is the shipped
partial example: it genuinely fills grammar, scope rules, placeholder policy,
prop order, and the incremental engine; it has **no** canonicalizer, oracle,
or typed-AST generator.

**Registry.** `register_pack` / `get_pack(dsl=None)` / `list_packs` live in
`dsl/pack.py`. `get_pack(None)` resolves through `active_dsl()` /
`SLM_GRAMMAR_DSL` exactly like `get_backend`, and backend aliases
(`openui-lark`, `openui-langcore`, `lark-openui`) resolve to the `openui`
pack. The pack registry does not duplicate the grammar-backend registry — the
`backend` slot references it.

## What got parameterized (the seams)

Default behavior is byte-identical everywhere (asserted by tests):

1. `dsl/production_codec.py` — `_prop_order(dsl=None)` (was a hardcoded
   `openui_prop_order.json` read; non-OpenUI ids resolve via the backend's
   new public `LarkFileBackend.prop_order()`), `_parse_program(source, dsl=None)`
   (was `get_backend("openui")` hardcoded), and `dsl=` threaded through
   `encode_openui` / `encode_choices` / `roundtrip_*` and the private
   encode/bindings chain. B1's `encode_choices`/`decode_choices` and
   `models/choice_tokenizer.py` keep working unchanged (they call the
   defaults).
2. `dsl/canonicalize.py` — the previously cosmetic `dsl=` parameter now
   reaches the codec (`encode_openui(..., dsl=dsl)`), not just the final
   re-validation.
3. `data/verify/stack.py` — the G1 grammar gate resolves its backend via
   `get_backend(dsl or "openui-lark")` instead of importing
   `OpenUILarkBackend` directly. Default deliberately stays the strict Lark
   CFG (G1 is a pure grammar check, separate from the lang-core schema gate
   G2); verdicts on fixtures are unchanged.
4. `data/progspec/generate.py` — `GeneratorConfig` gains `schema` /
   `prop_order` overrides; `None` resolves to the pinned OpenUI library
   schema and prop-order file as before.
5. `models/dsl_tokenizer.py` — `STRUCTURAL_TOKENS` now routes through the
   active grammar backend with the `dsl/openui_tokens.py` constant as
   fail-open fallback (mirroring `models/grammar.py`); the two sets are
   asserted identical for OpenUI, so vocab layout is unchanged.
6. `dsl/grammar/fastpath/engine.py` — `engine_for_dsl` consults the pack's
   `incremental_engine` slot first; the hardcoded alias list remains as the
   registry-free fallback.

## What is deferred (honestly)

- **No file moves.** The issue said "mostly moves"; in fact the ingredients
  are load-bearing modules imported all over `src/` and the honest F1 is
  edits + aggregation. Grouping OpenUI constants into a `dsl/packs/openui.py`
  layout is deferred to a dedicated `organize-repository` pass (`git mv` +
  same-change import updates).
- **`dsl/language_contract.py` stays OpenUI-pinned.** The contract-id
  singleton (pinned component schema hash) is a corpus-versioning concern;
  making it per-pack belongs with the first real second pack (F2).
- **The production codec's lexical layer is still OpenUI-shaped.** Statement
  regexes expect uppercase component heads, so `encode_openui(dsl="toy-layout")`
  parses (backend seam works) but cannot re-bind lowercase toy calls — which
  is exactly why toy-layout's `canonicalize` slot is an honest `None`, not a
  pretend implementation. A codec generalization (or per-pack codec slot) is
  F2 work.
- **F4 ontology variant** (graph-walk grammar, reasoner oracle) is contract-
  compatible by construction (Protocol-typed slots) but has no implementation.
- **Reward wiring**: `reward_label` is exposed and tested; making RL/reward
  harnesses read it from the pack (instead of their own labels) is F3 work.

### DSH3 operator capability

SLM-370 extends the same fail-closed slot pattern with optional
`operator_library`. The generic immutable registry and pure apply/dry-run/replay
boundary live in `dsl/operators/registry.py`; packs without the slot remain
unsupported. An operator-produced source passes the pack's normal parse,
static/schema oracle, scope, property-order, canonicalization, and round-trip
authorities before it can become a new immutable state. See
[dsh3-02-pack-operator-registry-20260723.md](dsh3-02-pack-operator-registry-20260723.md).

### DSH1 declared grammar capabilities

SLM-353 adds the optional `grammar_capability_authority` slot without adding a
registry. `GrammarCapabilityAdapterV1` resolves one `DslPack` and exposes only
that pack's declared start symbols, productions, terminals, grammar analyses,
fragment parser, canonical serializer, validator, scope policy, and completion
frontier. Missing declarations return typed `UNSUPPORTED` values, so partial
packs cannot appear complete.

OpenUI declares its authority from the strict Lark grammar inside its pack
wiring. The generic adapter imports no OpenUI implementation and has no
corpus-example fallback. See
[dsh1-01-grammar-capability-adapter-20260723.md](dsh1-01-grammar-capability-adapter-20260723.md).

SLM-354 extends that same authority object with exact production tracing,
finite containing-context witness candidates, and typed unsupported reasons.
`minimal_witnesses.py` remains pack-generic: it selects the lexicographic
minimum fully admitted candidate per reachable/productive alternative and
blocks every unexplained gap. OpenUI sources and lexer/semantic exclusions
remain behind `pack.py`. See
[dsh1-02-minimal-alternative-witnesses-20260723.md](dsh1-02-minimal-alternative-witnesses-20260723.md).

### DSH1 symbolic Harness task boundary

SLM-355 adds `harness_dsl/v1`, a separate closed grammar for CAP0 task intent.
The Harness parser owns only reserved operation/type framing, exact pack and
grammar-category symbols, digest artifact refs, declared runtime markers, and
the embedded payload boundary. It does not parse the target language itself.

After the outer prompt parses, generic code resolves the named `DslPack` and
requires its `fragment_parser` slot. OpenUI populates that existing slot with a
typed adapter around its document/statement/expression/lexical/node validator;
OpenUI-specific parsing remains behind `pack.py`. The symbolic-surface policy
then rejects comments, open strings/numbers, and undeclared identifiers or
runtime refs. Partial packs without typed fragment validation fail closed.

See
[dsh1-03-symbolic-harness-dsl-20260723.md](dsh1-03-symbolic-harness-dsl-20260723.md).

## End-to-end fixture run (executed, not hypothetical)

`tests/test_dsl/test_pack.py::test_end_to_end_fixture_run_through_pack_interface`
drives the full chain through `get_pack("openui")` only:

1. **generate** — `pack.corpus_generator(seed=0).generate(2)` (typed-AST,
   coverage-guided, N=2);
2. **verify** — `pack.oracle(source)` → both programs pass, tier Silver;
3. **canonicalize** — `pack.canonicalize` idempotent on both;
4. **scope** — `pack.scope_extractor` yields document/statement scopes;
5. **placeholder policy** — `pack.placeholder_policy.slot_contract` covers
   all extracted placeholders;
6. **train scratch** — `build_model(ModelBuildConfig(model_name="stub",
   grammar_dsl=pack.pack_id), records)`; the factory threads
   `grammar_dsl` into `set_active_dsl`, and one `forward` pass trains the
   stub (the full `scripts/train_model.py --model stub` CLI path is the same
   factory; the in-process run keeps the suite fast);
7. **eval** — the trained stub's output canonical-matches gold
   (`canonical_equal`) and passes the pack oracle;
8. **document** — this page.

A second test proves a **second pack resolves**: toy-layout backend parse +
scope extraction + slot contract + streaming engine on a toy program, and its
missing slots fail closed.

Recipe for a full-size rerun (identical interface, bigger N, real model):

```bash
NODE_OPTIONS= .venv/bin/python - <<'EOF'
from slm_training.dsl.pack import get_pack
pack = get_pack("openui")            # or SLM_GRAMMAR_DSL=...
gen = pack.corpus_generator(seed=0)
specs = gen.generate_until_covered().programs
assert all(pack.oracle(s.canonical_openui).ok for s in specs)
EOF
NODE_OPTIONS= .venv/bin/python -m scripts.train_model --model stub --steps 200 ...
```

## Run metadata

| Field | Value |
| --- | --- |
| Kind | contract formalization + seam refactor (no training run) |
| Date | 2026-07-17 |
| Branch / base | `claude/dsl-training-scope-optimization-ea7h3r` @ `bc81d1f` (includes B1 choice codec) |
| Checkpoint produced | n/a |
| Eval suite | n/a (behavior-preserving; suites below) |
| GPU / wall time | n/a (CPU test suites only) |
| Seeds | generator seed 0 in the e2e fixture test |
| JSON mirror | n/a (pure refactor; measured section below) |

## Measured: suites before / after

`NODE_OPTIONS= .venv/bin/python -m pytest tests/test_dsl tests/test_models tests/test_data -q`

| | passed | skipped | deselected |
| --- | --- | --- | --- |
| before (bc81d1f) | 435 | 4 | 3 |
| after | 454 (435 + 19 new pack tests) | 4 | 3 |

No pre-existing test changed outcome; the 19 new tests are
`tests/test_dsl/test_pack.py`.
