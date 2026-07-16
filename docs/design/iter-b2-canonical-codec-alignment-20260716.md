# Iteration — B2 canonical-space train/decode alignment audit (2026-07-16)

Machine-readable summary:
[`iter-b2-canonical-codec-alignment-20260716.json`](iter-b2-canonical-codec-alignment-20260716.json)
· Linear SLM-22 · no training run, no checkpoint — codec audit + fix only.

## Question

Does any many-to-one collapse applied by the deterministic production decoder
diverge from the training targets, recreating the E225-class train/decode
mismatch (`iter-e226-honest-compiler-policy-20260716.md`)? Training loss for
the X-series is computed on `encode_openui` token streams; the same codec's
`decode_productions` is the deterministic decoder at eval. If the stream is not
a fixed point of encode∘decode, the model is trained toward targets the decoder
cannot reproduce.

## Method

Sweep over both committed fixture corpora (`train_seeds.jsonl` +
`test_seeds.jsonl`, n=36 records, CPU only) checking three invariants:

1. **Token idempotence** — `encode(decode(encode(src))).tokens == encode(src).tokens`.
2. **Lang-core canonical equality** — resolved root AST (statementId erased)
   identical between `src` and `decode(encode(src))`, both parsed through the
   official `@openuidev/lang-core` bridge. The official serializer preserves
   binder names, so alpha-renamed round-trips are compared on resolved
   structure, not serialized strings.
3. **DSL-native tokenizer idempotence** — `DSLNativeTokenizer.canonicalize`
   fixed point with fresh per-example symbol tables.

## Measured results

| Check | Before fix | After fix |
| --- | --- | --- |
| Token idempotence failures | 3 / 36 | 0 / 36 |
| Lang-core canonical-equality failures | 2 / 36 | 0 / 36 |
| DSL-native tokenizer idempotence failures | 0 / 36 | 0 / 36 |

Four divergences were found and fixed in `src/slm_training/dsl/production_codec.py`:

1. **Dead-binding root mislabel.** The encoder emitted unreferenced statements
   *after* root while the decoder names the *last* stream statement `root`;
   `root = Card(..) / orphan = Button(..)` decoded with root bound to the
   Button. Statements are now ordered dependencies-first with root always last.
2. **Name-sensitive statement order.** `_statement_order` discovered refs with
   a regex over surface text, so binder names occurring inside string literals
   (e.g. binder `hero` and literal `":smoke.hero.kicker"`) changed the stream
   order. After the decoder's alpha-renaming removed those accidental matches,
   re-encoding produced a different token stream than the training target.
   Order is now derived from the parsed statement ASTs and is invariant under
   binder renaming.
3. **Children-first prop emission.** `_encode_component_props` emitted
   `children` before the declared positional prop order, so decoded surfaces
   reparsed with values in the wrong props — `Modal(":t", true, [body])`
   round-tripped to a program whose `title` was the children array. Emission
   now follows the declared positional order exactly (absent middle props pad
   with `null`; unknown props fail closed).
4. **v0.5 decode reorder.** `_decode_v05` moved root to the first output line
   while the v0.5 encoder preserves source statement order, breaking
   idempotence for state-first programs. The decoder now preserves stream
   order.

Forward references — including references to root, which is now always the
final statement — raise `ParseError` at encode instead of emitting a stream
that decodes to a different program.

## Regression tests

Beside the existing codec tests (`tests/test_dsl/test_production_codec.py`):
dead-binding root fidelity, Modal positional-prop round-trip, forward-ref
fail-closed, fixed-point checks for targeted sources and the full fixture
corpora, and the lang-core bridge canonical-equality test (skips when the
bridge is not installed). `DSLNativeTokenizer` idempotence is pinned in
`tests/test_harnesses/model_build/test_dsl_tokenizer.py`.

Suites run green: `tests/test_dsl`, `tests/test_models`,
`tests/test_harnesses/model_build` (275 passed), plus `scripts.repo_policy`,
`.githooks/check-changed`, `ruff`, `git diff --check`.

## Honest caveats

- Fixture-corpus wiring evidence only — no checkpoint, matrix row, or ship
  claim. Meaningful-parse impact must be measured by rerunning the X-series
  matrices on post-fix streams.
- Production token streams changed for programs hit by the divergences, so
  X-series checkpoints trained on pre-fix streams are not comparable without
  retraining.
- The eval-side exact-match asymmetry remains: `eval_runner._tree_match`
  compares codec-canonical predictions against echo-normalized gold
  (`lark_backend.serialize` is an input echo, not a canonicalizer). That is
  D2's charter (e-graph canonicalizer for canonical exact-match eval, SLM-30)
  and was deliberately not changed here — changing eval metrics belongs with
  `honest-ship-eval` review.

## Conclusion

`encode_openui` token streams are now fixed points of the deterministic decode
collapse and canonically equal (resolved AST modulo statementId) through the
official lang-core bridge across both fixture corpora. B1 (choice-sequence
codec, SLM-42) can build on a codec whose training targets and decode-time
collapse agree.
