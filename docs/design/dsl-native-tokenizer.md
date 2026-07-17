# DSL-native output tokenization (V5)

Design note for replacing textual / string-piece tokenization on the
**output** side with a reversible compiler-derived intermediate representation.

Companion: [quality-experiment-matrix.md](quality-experiment-matrix.md) (E40–E46),
[research-correction-critics.md](research-correction-critics.md) (critics consume
statement/symbol remask targets), [research-lineage.md](research-lineage.md).

## Claim

You do **not** want a statistically learned subword tokenizer for OpenUI
generation. You still need a finite alphabet and learned embeddings, but that
alphabet should come from the DSL’s lexer/grammar rather than BPE /
SentencePiece / corpus-grown string pieces.

Asymmetric interface:

```text
natural-language tokenizer/encoder
        → shared latent (context tower)
        → DSL-native diffusion decoder
```

In this repo that maps to:

| Side | Implementation |
| --- | --- |
| Input / context | Existing scratch `OpenUITokenizer` (prompt words) or HF SmolLM2 tokenizer |
| Output / denoiser | New [`DSLNativeTokenizer`](../../src/slm_training/models/dsl_tokenizer.py) |

Config flag: `output_tokenizer="compositional" | "lexer"` (default remains
compositional so V2/V3 checkpoints stay bit-identical).

## Why the v2 compositional tokenizer is not enough

[`OpenUITokenizer`](../../src/slm_training/models/tokenizer.py) already avoids BPE,
but still:

1. Mixes prompt English into the **output** vocab (`from_records` builds on
   `prompts + openui`).
2. Spells placeholders as compositional subtokens (`"`, `:`, `hero`, `.`, `title`, `"`),
   inflating sequences (~p95 112 on fixtures) and making remask target arbitrary spans.
3. Models whitespace as tokens.
4. Couples grammar terminals to ids via heuristic vocab scans
   (`token_map.allowed_id_set`).

## Output channels

### Fixed grammar vocabulary

One id per structural terminal (`=`, delimiters, object/expression operators,
`NL`, `.`), each OpenUI component keyword (`Stack`, `Card`, …), runtime builtin
(`Query`, `Mutation`, `Action`, `@Run`, …), `true`/`false`/`null`, typed literal
openers (`LIT_STR`, `LIT_NUM`, `LIT_END`), and closed string atoms
(`STR:column`, `STR:row`, …).

### Dynamic symbol table (pointer / copy)

Per example, binders are alpha-renamed to `<BIND_j>`, state names to
`<STATE_j>`, and placeholders from the slot-contract inventory bind to `<SYM_i>`:

```text
TextContent( <SYM_0> )     # ← ":hero.title"
```

Overflow beyond `sym_slots` / `bind_slots` falls back to the literal/byte
channel (counted, never hard-fails).

### Typed literal channel

Non-inventory strings and numbers: `LIT_STR` / `LIT_NUM` + printable-byte tokens
+ `LIT_END` (ByT5-style robustness without putting every identifier permanently
in the vocab).

### Macro tokens (C3, tokenizer v3)

64 reserved `<MACRO_i>` rows appended after the `<STATE_k>` pool
(`DSL_TOKENIZER_VERSION = 3`; all prior ids unchanged). A macro binds one id to
a fixed span of **fixed-vocabulary** token ids only
(`MACRO_EXPANDABLE_KINDS = {struct, component, builtin, lit, byte}` — never
`NL` and never the dynamic `<SYM_i>`/`<BIND_j>`/`<STATE_k>` pools, so macros
are alpha-independent by construction and the pitfall of context-sensitive
alpha-equivalence hashing never arises). The expansion table is mined offline
by [`data/macro_induction.py`](../../src/slm_training/data/macro_induction.py):
greedy iterative MDL over canonicalized training sources, picking the n-gram
(length 2..8) with the best `net_gain = freq * (len - 1) - len` until gains
fall below threshold (Stitch/LILO-style compression, deterministic and
lossless — no learning, no anti-unification).

`encode()` greedily substitutes table spans (longest-first) after lexing;
`decode()` splices expansions back before any other processing, so everything
downstream (verifier, grammar masks, evals) sees the expanded program. The
table is persisted inside the tokenizer JSON — train and decode can never
disagree. Fail-closed rules: `set_macro_expansions` rejects spans containing
dynamic-pool or unknown tokens; `decode` drops a `<MACRO_i>` id that has no
table entry rather than emitting the raw sentinel. Enabled per-run via
`macro_tokens=true` (config → factory → `TwoTowerModel.from_records`), and the
diffusion adapter gains a `macro_substitution` corruption policy that masks
whole macro tokens (one id = one block edit).

### Surface-identifier arm (C4, comparison-only)

`symbol_anonymization=False` (encode flag, threaded from `TwoTowerConfig`)
routes binder and state names through the byte channel verbatim instead of
the `<BIND_j>`/`<STATE_k>` pools; placeholders keep `<SYM_i>`. This exists
solely as the surface arm of the C4 names-disappear comparison
(arXiv:2510.03178) — it is not a shipping representation. Decode needs no
special handling (byte runs already reassemble into surface pieces), and the
round trip is exact without a symbol table. Fail-closed: incompatible with
`bind_encoding="relative"`, `macro_tokens=True`, and `grammar_constrained`
decode (the NAME gate admits only `<BIND_j>` ids), all of which raise. On the
fixture corpus the surface encoding is ~1.72× longer.

### Factorized embeddings (Stage 2)

Optional `E_tok[id] + E_kind[kind(id)] + E_pos[i]` in
[`DenoiserTower`](../../src/slm_training/models/blocks.py) behind
`factorized_embeddings=True`.

Kind metadata
(`special|struct|component|builtin|sym|bind|state|lit|byte`) is serialized in
the tokenizer JSON and is the authority for grammar masks. The added v2 kinds
extend the factorized kind table without renumbering the original seven kinds.

## Interaction with MaskGIT / remask / critics

| Lever | Config | Role |
| --- | --- | --- |
| Mixed statement masking | `mask_pattern=mixed` | Train with `M_random ∪ M_statement` (NL-delimited spans) |
| Statement remask | `remask_span=statement` | Expand confidence remask to the enclosing statement |
| Exact grammar masks | automatic for lexer tokenizer | `token_map` uses kinds instead of heuristic scans |
| Template fill | existing E20 | Marks `<SYM>` / `<BIND>` positions via kinds |

This is Stage 1–2 of the feedback progression. **Not** done here (V6
candidates):

- Production-rule / parser-action sequences (alignment-unstable under masked diffusion).
- Full typed AST-slot / graph diffusion.
- Tree-annotated flat canvases with relative tree attention (Stage 3).
- Latent falsification MoE critics (E34) — but statement/symbol remask targets
  are the stable units those critics should emit responsibility over.

## Experiments

See V5 matrix **E40–E46** in [quality-experiment-matrix.md](quality-experiment-matrix.md).

Length diagnostic:

```bash
python -m scripts.diagnose_tokenizer --fixtures
```

Fixture seeds (n=20): compositional mean **72.6** tokens → lexer+symtable **46.3**
(ratio **0.64**); fixed output vocab **296**, context vocab stays corpus-sized.

## Recommended default going forward

After proving quality (V5 E46 / V6 E53):

```text
General tokenizer for input
+ lexer/grammar-native output vocabulary
+ dynamic per-example symbol table
+ byte/copy fallback for literal content
+ factorized kind embeddings
+ CFG-constrained LTR/MaskGIT decode
+ critic-guided structural remasking (E33)
+ CoRe-lite context-robust remask (E50)
+ T2M remask→mask discipline (E51)
+ honest inventory-in-prompt (E35) + slot-aware trust (E52)
```

Do **not** jump to pure production-rule sequences or graph diffusion until the
critic / remask stack is proven on this representation. The parallel **X matrix**
(`grammar_diffusion`) explores production codecs under honest inventory
contracts (E54 / X2–X7).

## V8 request-conditioned symbols (implemented, experiments unrun)

`GenerationRequest.runtime_symbols` distinguishes `alpha_binder`,
`fresh_binder`, `external_entity`, and `state` roles while preserving the legacy
`slot_contract`. Symbol-table serialization is v3 and reads v2 tables by
deterministically reconstructing metadata; reserved token IDs do not move.

`runtime_symbol_features=surface|role_gated` pools existing byte embedding rows
for request-visible metadata and applies the resulting per-example delta at both
input embedding and tied output projection. `role_gated` suppresses surface
features for binders, retaining alpha-renaming invariance, while entities and
states remain name-aware. `none` is the legacy path. Optional training slot
permutation preserves root binder slot zero.

Semantic masks conservatively hide undeclared entity/state rows but leave binder
rows writable. The constraint graph reuses V7 clustering and ordered verification;
it adds statement adjacency, repeated symbol identity, and matched delimiter edges
in `grammar` or `hybrid` mode. This is not a complete type/scope solver.

Primary V8 diagnostics are alpha-renaming invariance, slot-permutation robustness,
entity exact-copy accuracy, active-symbol recall, false hard-prunes, graph
conflicts/recovery, active canvas tokens, parse/AST equivalence, and unchanged
honest ship gates. No V8 quality or latency result exists yet.
