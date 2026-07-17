# C1 — Relative-index (De Bruijn) references (2026-07-17)

Representation primitive, not a train/ship run. Code:
[`dsl/production_codec.py`](../../src/slm_training/dsl/production_codec.py)
(`to_relative_refs` / `from_relative_refs`, opt-in `relative_refs=` on
`encode_openui` / `encode_output` / `roundtrip_openui` / `ProductionCodec`).
Linear SLM-25.

## What and why

The grammar-native production codec already emits statement references as
integer slot pointers (`&i`, an absolute index into canonical statement order),
so the model never invents a binder name — the "names disappear" defense
(C4 / arXiv:2510.03178) holds at the reference site. C1 goes one step further:
make the reference **scope-relative** instead of absolute.

A use-site ref `&i` inside the statement at canonical index `cur` becomes
`~{cur - i}` — the signed distance, in canonical statement order, from the use
back to the binder's definition (a De Bruijn index over the flat statement
scope, motivated by the neural binding problem, Greff et al. 2020). The point is
**translation invariance**: inserting or deleting an unrelated earlier statement
renumbers every absolute slot after it, but leaves the local `def→use` distance
unchanged for refs that do not straddle the edit. A diffusion edit near one
binder therefore does not perturb the token identity of references elsewhere in
the program — the property absolute indices lack.

## Relationship to D2

D2's canonicalizer renames binder **definitions** to a De Bruijn-style
`v0, v1, …` pool (context-insensitive alpha-renaming of *names*). C1 is the
complementary half: it re-expresses **references** as relative distances. Both
externalize binding, but D2 normalizes the *definition* surface while C1
normalizes the *use* surface. They compose — canonical statement order is the
shared coordinate system both rely on.

## Mechanism (honest framing)

Two pure token-stream transforms, no change to any existing encode/decode
internals:

- `to_relative_refs(tokens)` — single left-to-right pass; track the current
  statement index by counting statement markers (`=` for documents, the five
  v0.5 sigils for the runtime surface); rewrite each `&i` to `~{cur-i}`.
- `from_relative_refs(tokens)` — the inverse, **verifier-enforced**: a delta
  resolving to a negative (undefined) statement index raises `ParseError`. The
  absolute-ref decoders already range-check the upper bound against the decoded
  binder set, so legality is a property of the parser, not something the model
  must learn.

Encode is opt-in (`relative_refs=True`); decode auto-detects the `~` sigil and
restores absolute indices before dispatch, so a relative stream round-trips
through the unchanged document / v0.5 / fragment decoders. Default paths and the
fixed vocabulary are untouched — no checkpoint migration.

Document refs run backward in canonical order (referents precede users in the
topological statement list), so their deltas are `>= 1`. The v0.5 runtime
surface allows forward references, so its deltas are signed.

## Forward-compatibility (F2 / GraphQL, pattern DSL)

The relative index is defined over "the enclosing statement scope's canonical
order." For OpenUI 0.2.x that scope is the flat top-level statement list (state /
queries deferred, so the binding surface is small — the honest reason C1 is a
representation change proven by round-trip, not an accuracy win yet). For a
future GraphQL pack the scope splits: intra-document fragment/variable bindings
stay relative, but schema symbols resolve against the introspection schema (an
absolute symbol table, not a relative distance). The transform takes the marker
set as its only scope parameter, so a pack supplies its own statement-boundary
predicate without touching the ref arithmetic.

**Caveat (arXiv:2401.02948).** Relative indexing gives translation invariance,
not context-sensitive subterm identity. Two structurally identical references in
different contexts still get the same delta; detecting genuinely shared subterms
is C3's job (macro induction), which must not assume equal deltas imply shared
context.

## Verification

- `tests/test_dsl/test_production_codec.py`: deltas emitted (document deltas
  `>= 1`), document and v0.5 round-trip through the parser, relative and absolute
  streams decode identically, `from∘to` is the identity, translation-invariance
  across an inserted statement, illegal (out-of-scope) delta rejected.
- 28 codec tests green (B2 fixed-point suite + the C1 additions above).
- No checkpoint, no scoreboard, no ship claim.

## Deferred

The E-row on the quality matrix (meaningful-parse primary, per the issue's
verify clause) is left for a matrix run under `scripts/run_quality_matrix.py`
with `relative_refs` enabled on the production codec — the representation and its
round-trip guarantees land here; the accuracy delta is measured when the B-series
choice-sequence codec (B1, SLM-42) is wired, since that is the training path the
relative refs feed.
