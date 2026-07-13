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

One id per structural terminal (`=`, `(`, `)`, `[`, `]`, `,`, `NL`, `.`),
each OpenUI component keyword (`Stack`, `Card`, …), `true`/`false`/`null`,
typed literal openers (`LIT_STR`, `LIT_NUM`, `LIT_END`), and closed string atoms
(`STR:column`, `STR:row`, …).

### Dynamic symbol table (pointer / copy)

Per example, binders are alpha-renamed to `<BIND_j>`; placeholders from the
slot-contract inventory bind to `<SYM_i>`:

```text
TextContent( <SYM_0> )     # ← ":hero.title"
```

Overflow beyond `sym_slots` / `bind_slots` falls back to the literal/byte
channel (counted, never hard-fails).

### Typed literal channel

Non-inventory strings and numbers: `LIT_STR` / `LIT_NUM` + printable-byte tokens
+ `LIT_END` (ByT5-style robustness without putting every identifier permanently
in the vocab).

### Factorized embeddings (Stage 2)

Optional `E_tok[id] + E_kind[kind(id)] + E_pos[i]` in
[`DenoiserTower`](../../src/slm_training/models/blocks.py) behind
`factorized_embeddings=True`.

Kind metadata (`special|struct|component|sym|bind|lit|byte`) is serialized in
the tokenizer JSON and is the authority for grammar masks.

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
