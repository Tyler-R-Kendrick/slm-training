# B1 — Choice-sequence codec (semantic decisions only) (2026-07-17)

Representation primitive, not a train/ship run. Code:
[`dsl/production_codec.py`](../../src/slm_training/dsl/production_codec.py)
(`to_choice_stream` / `from_choice_stream`, opt-in `choice_stream=` on
`ProductionCodec`). Linear SLM-42.

## What and why

The endpoint of the "remove non-lexical symbols + deterministic decoder"
hypothesis: the model should predict only semantic decisions; every token the
grammar forces should be reconstructed, not learned. Honest accounting of where
that collapse actually lives:

- **Most of it already happened.** The production codec (`encode_openui`)
  collapsed the surface syntax — parens, commas, quotes, positional-prop
  scaffolding — into typed tokens long before B1. On the committed train
  fixtures the production stream is already **0.61×** the surface-lex stream.
- **B1's increment** removes the residual grammar-forced *framing*: the `=`
  statement markers in document streams (top-level expressions are
  self-delimiting, so the marker carries zero bits) and the `;` terminators in
  v0.5 streams (the next typed statement marker is the boundary). The typed
  v0.5 markers (`r=`, `$=`, `q=`, `m=`, `a=`) stay — statement *kind* is a
  genuine semantic choice.

Measured on the 20 committed train seeds (deterministic count, no model claim):
surface-lex 869 tokens → production 529 (**0.61×**) → choice 437 (**0.83× of
production, 0.50× of surface**). Every remaining token is a semantic decision:
which component, which slot filler, which literal, which ref, where a variable-
arity frame ends. This is the representation whose entropy the E1
bits-per-semantic-decision metric is meant to measure.

## Mechanism (honest framing)

Two pure token-stream transforms, mirroring C1's shape:

- `to_choice_stream(tokens)` — filter `=` (document) / `;` (v0.5) from the
  stream; fragment streams pass through unchanged.
- `from_choice_stream(tokens)` — reconstruct the framing deterministically:
  document statement boundaries by walking the self-delimiting expression
  grammar (`+Comp…-` and `[…]` frames balance; single tokens are complete
  expressions), v0.5 boundaries from the typed markers. Unbalanced or truncated
  frames raise `ParseError` — fail closed, per the honest-compiler policy.

`ProductionCodec(choice_stream=True)` applies the transform on encode and
inverts it on decode (inside the existing `ParseError → ""` guard, so masked/
truncated streams degrade exactly as before). **Not self-describing**: unlike
C1's `~` sigil, a choice stream is distinguishable from a production stream only
by configuration, so decode never sniffs — the flag is part of the codec
identity, and a checkpoint's codec config must record it.

Composes with C1 (relative refs): refs are converted first — the De Bruijn
delta arithmetic counts statement markers — then the framing is elided. On
decode the framing is reinserted before `decode_productions` auto-restores
absolute refs, so the deltas resolve to the same statement indices as at
encode time. `ProductionCodec.build(relative_refs=True, choice_stream=True)`
is the combined configuration.

## Ship blocker (B2)

Per the issue: shipping any model trained on this representation is **blocked
by B2** (canonical-space training consistency, SLM-22) — loss must be computed
in the same canonical space the deterministic decoder emits, or we recreate the
E225 train/decode mismatch. B2's fixed-point tests (`encode(decode(tokens)) ==
tokens`) landed on `main`; the choice-stream fixed-point test here extends the
same contract to this representation.

## Verification

- `tests/test_dsl/test_production_codec.py`: marker elision counts, `from∘to`
  identity on both surfaces (document + v0.5) and the full committed fixture
  corpus, the issue's `choices → serialize → parse → choices` fixed point,
  composition with relative refs, `ProductionCodec` end-to-end round-trip, and
  fail-closed rejection of an unbalanced frame.
- 36 codec tests green.
- No checkpoint, no scoreboard, no ship claim.

## Deferred

E-rows on the quality matrix (meaningful parse primary, E2 semantic-density
gates) require training runs against this codec configuration — deferred to a
matrix campaign alongside B3's capacity ladder, which consumes exactly the
surface-vs-choice comparison this codec makes possible. The B2 ship blocker
must clear first.
