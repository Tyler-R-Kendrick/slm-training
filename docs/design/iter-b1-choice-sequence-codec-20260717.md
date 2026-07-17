# B1 (SLM-42): choice-sequence codec — model predicts only semantic decisions

Date: 2026-07-17 · Track: B1 · Linear: SLM-42 · Blocked-by B2 (SLM-22, closed)

## What shipped

A third output representation beside the compositional and lexer-native
tokenizers: a **pure grammar-choice stream**. The model predicts only semantic
decisions (which production, which slot filler); ALL non-lexical surface syntax
is reconstructed by a deterministic detokenizer routed through the official
`@openuidev/lang-core` serializer, failing closed on invalid reconstruction.

- `src/slm_training/dsl/production_codec.py` — `encode_choices()` /
  `decode_choices()` / `roundtrip_choices()`; v0.5 expression Pratt parser
  (`_V05ExprParser`, mirrors `openui.lark` precedence), pre-order choice
  emission, precedence-aware renderer, serializer-routed canonicalization.
- `src/slm_training/models/choice_tokenizer.py` — `ChoiceTokenizer`
  (sidecar kind `choice_codec`), grammar-closed deterministic vocabulary
  (681 ids), framed byte channels for open literals/keys, fail-closed decode.
- Wiring: `--output-tokenizer choice` (`scripts/train_model.py`),
  `ModelBuildConfig`/`TwoTowerConfig` passthrough, `TwoTowerModel.from_records`
  branch, `_load_any_tokenizer` sidecar dispatch, choice branches in
  `_encode_openui`/`_decode_openui`/placeholder-id/fidelity-loss sites, and a
  v1 bypass of the surface-DFA token gate during generation.
- Matrix: `_v12_experiments()` → **E262** (`qx_e262_b1_choice_codec`),
  matched against E255 (identical diffusion masking + non-LTR MaskGIT decode,
  differing only in output representation).
- Tests: `tests/test_dsl/test_choice_codec.py` (18 tests),
  `tests/test_models/test_choice_tokenizer.py` (9 tests).

## Canonical-space semantics (B2 alignment)

The choice stream lives in **canonical space**: `encode_choices` first
collapses the input to the official serializer form. This is what keeps the
B2-shaped laws true for every valid input (pinned in
`tests/test_dsl/test_choice_codec.py`, mirroring
`tests/test_dsl/test_canonical_alignment.py`):

1. **Loss space is a fixed point of decode**: choices → decode(serialize) →
   re-encode → identical choices.
2. **Decode of any stream is a lang-core serializer fixed point** (decode is
   routed through `validate(text).serialized`; fail-closed `ParseError` when
   the reconstruction is invalid).
3. **Canonical equality modulo the documented alpha-renaming**: resolved AST
   of decode output equals the resolved AST of the input's canonical form,
   modulo positional binder renaming (`root`, `v*`, `q*`, `m*`, `a*`, `$s*`).

Inherited serializer collapses (deliberate, verified): statement reordering,
dead-binding pruning, trailing-`null` prop pruning, redundant-paren removal
(the serializer re-associates `"" + (1 + 2 * 3)` to `"" + 1 + 2 * 3`; the
codec adopts the serializer's AST as authoritative). For already-canonical
inputs, law 3 reduces to plain alpha-modulo equality with the source.

## Per-sigil keep/drop table

| Token | Verdict | Rationale |
| --- | --- | --- |
| `+Comp` | keep | production choice: which component / call head |
| `*Builtin` | keep | production choice: which builtin action/aggregate |
| `-` (close) | keep | arity decision — prop order declares order, not arity |
| `[` / `]` | keep | shape decision — grammar admits scalar or list anywhere |
| `{` / `}` | keep (new) | object-literal shape + entry count (v0.5) |
| `@i` | keep | slot filler choice |
| `&i` | keep | statement reference choice |
| `$@i` | keep | state reference choice |
| `#lit` | keep | literal filler choice |
| `^dir` | keep | enum filler choice |
| `n:name` | keep | object key / unbound identifier — irreducible content choice |
| `.name` (new) | keep | fused member-access choice (that it happens + which member) |
| `o:<op>` (new) | keep | operator choice; operands positional, so no parens |
| `r= $= q= m= a=` | keep (v0.5) | statement-production choice (`$x = 0` vs `x = 0` not derivable from RHS; root position is source order); `q=/m=/a=` derivable from RHS head but kept for local alpha-naming |
| `=` (STMT) | **drop** | structural path: statements are self-delimiting expressions |
| `;` (EOL) | **drop** | grammar-forced: statement ends when its expression completes |
| `!v0.5` | **drop** | derivable: stream starting with a statement marker is v0.5 |
| `p:punct` | **drop** | all surface punctuation reconstructed from the grammar |

## Measured semantic density (E2)

Corpus: 36 committed fixture seeds (`train_seeds.jsonl` + `test_seeds.jsonl`),
unigram description length via `evals/semantic_bits.py`
(`compare_representations`, now with a `choice` stream and
`categorize_choice`). JSON mirror:
[iter-b1-choice-sequence-codec-20260717.json](iter-b1-choice-sequence-codec-20260717.json).

| Stream | decisions | alphabet | total bits | bits/decision |
| --- | ---: | ---: | ---: | ---: |
| surface (lexemes) | 1535 | 244 | 8368.0 | 5.451 |
| production | 1019 | 54 | 4391.9 | 4.310 |
| **choice** | **842** | **53** | **3713.2** | 4.410 |

Category collapse (production → choice, same corpus):
`structural 506 → 0`, `punct 0 → 0`, `name 0 → 0`; the choice stream keeps
`arity 329` (close/list decisions — genuine variable-arity choices, no longer
mislabeled "structural"), production 177, reference 141, slot 118, direction
40, literal 37.

Ratios: `surface_to_production_bit_ratio` **1.905** →
`surface_to_choice_bit_ratio` **2.254**; `production_to_choice_bit_ratio`
**1.183** (−17.4% decisions, −15.5% total bits vs the production stream).
Irreducible remainder on v0.5 programs: `name` (object keys) and `member`
tokens — genuine key choices, counted as decisions, not surface residue
(pinned by `test_choice_categories_collapse_surface_residue`).

## E262 matrix row + fixture wiring smoke

E262 is **registered** in `scripts/run_quality_matrix.py` (`--matrix v12`)
but **unrun as a matrix row**: this environment has no
`outputs/data/eval/v1` suite corpus (rico/ood suites absent), and a
production verdict needs the standard budget anyway.

A direct fixture-scale CPU smoke of the same harness path
(`model_build.train` + `evaluate_suites`, fixture seeds: 24 train / 16 smoke)
verifies the wiring end-to-end. **Wiring evidence only — honest zeros; this
budget cannot support any quality claim in either direction:**

| Run | Config | parse | fidelity | notes |
| --- | --- | ---: | ---: | --- |
| choice 120 | d64, random mask, 120 steps | 0.0 | 0.0 | train 9.2s; eval 0.9s (fails closed fast) |
| lexer 120 (control) | d64, random mask, 120 steps | 0.0 | 0.0 | train 5.9s; eval 149.7s (grammar repair) |
| choice 800 | d128, random, 800 steps | 0.0 | 0.0 | |
| choice 2500 | d128, random, 2500 steps | 0.0 | 0.0 | train-set memorization also 0/8 |
| choice 1200 | d128, diffusion, 1200 steps | 0.0 | 0.0 | length buckets over-allocate (32 vs gold 20) |

Failure mode (inspected raw canvases): the model learns the gold prefix
(`+TextContent @0 - …` matches for ~11 tokens) but cannot place `<eos>`
on an over-allocated canvas; excess masked positions fill with garbage that
breaks the prefix-free stream, and the detokenizer fails closed to `""`
(honest parse 0 instead of lenient garbage text). The matched lexer control
scores the same 0.0 at this scale, so this is a fixture-budget capability
wall, not a choice-codec regression. Length-faithful decode (tight length
buckets or an explicit choice-stream end decision) is the first follow-up.

Run metadata:

| Field | Value |
| --- | --- |
| device | CPU (scratch context tower) |
| steps | 120 / 800 / 1200 / 2500 (see table) |
| backend | scratch; MaskGIT parallel decode; grammar_ltr_primary=False |
| matrix | v12 row E262 registered; matrix itself unrun (no eval corpus here) |
| n | smoke 16 (fixture test seeds); train 24 (fixture train seeds) |
| honesty | honest (no gold leak; fail-closed decode; slot contract in context for d128 runs) |
| gate | none claimed; parse primary honest 0.0 at fixture scale |

## Tradeoffs and caveats

- **Grammar-closed vocab**: components, operators, markers, slot/ref/state
  pools, prop-name keys, and a fixed literal pool are enumerated from the
  grammar (not the corpus). Unseen components fail closed to `<unk>` (and are
  rejected by the parser before that); free-form string/number literals and
  object keys/member names outside the pools use framed byte channels
  (`LIT_STR`/`LIT_NUM`/`NAME_STR`/`MEMBER_STR` + printable-ASCII bytes) —
  content there is a genuine semantic decision, so it is spelled, not dropped.
  Non-ASCII content fails closed.
- **Constrained decode**: v1 bypasses the surface-DFA token gate when the
  output tokenizer is the choice codec (choice ids are not surface lexemes,
  so the lexer-kind content-id masks cannot admit them token-by-token). The
  code paths are guarded — nothing crashes; validation moves entirely to the
  fail-closed detokenizer. Follow-up: a choice-native legal-decision gate via
  `OpenUIIncrementalEngine.next_terminals()` / `gold_compiler_decisions()`.
- **Fragment outputs** (`output_kind != document`) are not supported by the
  choice codec in v1; documents only.
- **Environmental note**: the post-train loss-suite report requires the
  AgentV Node SDK; `npm ci` at the repo root was needed in this environment
  (pre-existing `tests/test_evals/test_agentv.py` failures cleared by it).
- **MODEL_CARD is intentionally NOT updated**: no checkpoint was created or
  promoted by this iteration (fixture smoke artifacts live only in the session
  scratchpad).
