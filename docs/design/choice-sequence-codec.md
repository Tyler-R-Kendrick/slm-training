# Choice-sequence codec (B1, SLM-42)

The endpoint of the "remove non-lexical symbols + deterministic decoder"
hypothesis: the model predicts **only semantic decisions**; every
reconstructible surface token is emitted by a deterministic detokenizer and
certified by the official lang-core serializer.

Code: [`dsl/production_codec.py`](../../src/slm_training/dsl/production_codec.py)
(`encode_choices` / `choices_to_productions` / `decode_choices` /
`choice_stats`). Tests: `tests/test_dsl/test_choice_codec.py`.

## The transform

The grammar-native production stream already removed binder names and most
surface syntax. The choice layer strips what remains reconstructible:

| Production token | Choice stream | Why |
| --- | --- | --- |
| `=` statement marker | *dropped* | a statement begins at every component open at nesting depth 0 — pure state |
| `]` list close / `-` component close | one generic `.` stop | at any point exactly one closeable scope is on top of the stack; the concrete delimiter is state, not choice |
| `+Comp @k &i ~±d ^dir #lit [` | kept verbatim | real decisions: which component, which slot filler, which referent, layout direction, literal, list-shaped prop |

`decode_choices` re-expands deterministically (a two-symbol scope stack),
then runs the existing `decode_productions` path — canonical statement
order, `v0…vn` binder pool, and official-serializer validation all reused.
Illegal streams fail closed (`ParseError` on over-closing or unclosed
scopes); a stop decision is a *choice about arity*, so it stays in the
model's vocabulary — but as one symbol, not two.

`choice_stats` supplies the E2 bits-per-semantic-decision inputs:
choice-token count vs production-token count vs surface atoms (the fixture
document: fewer decisions than production tokens, <0.5 decisions per
surface atom).

## Scope and deferrals (recorded)

- **v0.5 sidecar programs** (state/query/mutation) raise — the v0.5 marker
  stream mixes lexical fragments whose choice decomposition needs its own
  design pass.
- **Relative refs (C1)**: `~±d` tokens pass through the choice layer
  untouched; wiring `bind_encoding=relative` into `encode_choices` is a
  one-line composition once the production-codec relative transform (PR
  #277) lands.
- **Training half** — a tokenizer option beside
  `OpenUITokenizer`/`DSLNativeTokenizer` emitting choice streams as model
  targets, plus quality-matrix E-rows — is *ship-blocked by B2 (SLM-22)*:
  loss must be computed in canonical space before a choice-target model can
  be promoted (the E225 train/decode-mismatch lesson). The codec core lands
  first so B2's audit has the target representation in hand.

## Honesty

Codec-layer evidence only: round-trip identity (`choices → OpenUI →
choices`), detokenizer/production-decode equality, fail-closed properties,
and decision-surface stats. No model trained on choice targets, no E-row,
no ship claim.
