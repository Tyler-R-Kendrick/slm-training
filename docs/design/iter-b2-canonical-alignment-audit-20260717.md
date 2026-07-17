# Iteration — B2 canonical-space train/decode alignment audit, reopened (2026-07-17)

Linear SLM-22 (reopened) · no training run, no checkpoint — audit + property
tests only. Follow-up to
[`iter-b2-canonical-codec-alignment-20260716.md`](iter-b2-canonical-codec-alignment-20260716.md)
(PR #266), which covered `dsl/production_codec.py` only. The reopened scope
audits the two remaining codecs named by the issue —
`models/dsl_tokenizer.py` (`DSLNativeTokenizer`, lexer-native output path) and
`models/tokenizer.py` (`OpenUITokenizer`, compositional output path) — against
the official `@openuidev/lang-core` serializer (`validate(x).serialized`) as
the canonical form, per `docs/design/task-equivalence-eval.md`.

## Question

Does any many-to-one collapse applied at decode time diverge from training
targets on the lexer-native or compositional output paths, recreating the
E225-class train/decode mismatch? Canonical reference: the official lang-core
serializer, which normalizes whitespace (`name = Comp(a, b)`), double-quotes
strings, orders statements root-first then dependencies-first (post-order
DFS), drops trailing `null` props, prunes unreferenced bindings, and preserves
binder/state names.

## Method

CPU-only sweep over both committed fixture corpora (`train_seeds.jsonl` +
`test_seeds.jsonl`, n=36) plus targeted programs (dead-binding chains, Modal
bool/null positional padding, escaped string literals, binder names aliased
inside placeholder literals, v0.5 state/query/mutation/action), all through
the Node bridge. Committed training corpora were swept for
canonical-fixed-point status of document records. Invariants per codec:

1. **Serializer fixed point of decode** — for canonical input `x_c =
   validate(x).serialized`, the decoder/canonicalizer output `d` satisfies
   `validate(d).serialized == d`.
2. **Loss-space fixed point** — re-encoding the decoded surface reproduces the
   identical token/id stream.
3. **Canonical equality modulo alpha** — lang-core resolved root AST equal
   after erasing `statementId` and positionally renaming statement-bound
   names (the decoders' documented deterministic alpha-collapse; the official
   serializer preserves names, so string equality is not the right law for
   name-carrying v0.5 nodes).

## Measured results

| Check | production codec | DSLNativeTokenizer | OpenUITokenizer |
| --- | --- | --- | --- |
| Serializer fixed point of decode (36 fixtures + 6 targeted) | 0 failures | 0 failures (`canonicalize`) | identity codec (trivial) |
| Loss-space token/id idempotence | 0 failures | 0 failures | 0 failures (exact surface) |
| Lang-core canonical equality mod alpha | 0 failures | 0 failures | trivial |
| `canonicalize` idempotence | n/a | 0 failures | n/a |

Corpus sweep (document records, canonical-fixed-point of `record.openui`):

| Corpus | docs | divergent | verdict |
| --- | --- | --- | --- |
| `train_seeds.jsonl` + `test_seeds.jsonl` (raw seeds) | 36 | 6 | statement order only; collapsed by the seam (below) |
| `outputs/data/train/scope_graded_v1` | 774 | 26 | all 26 are `scope_identity_*` echo records with `meta.preserve_verbatim` — intentional by family design |
| `resources/data/train/remediated` | 585 | 0 | canonical |
| `resources/data/train/e218_schema_normalized_judge_v5` | 480 | 0 | canonical |
| `resources/data/train/e230_diverse_judged_roots_v2` | 65 | 0 | canonical |
| `resources/data/eval/remediated` (smoke/held_out/adversarial) | 12 | 0 | canonical |

## Findings

**No unintentional train/decode canonical divergence remains.** Per codec:

1. **`dsl/production_codec.py`** — aligned (post-#266). Newly verified beyond
   #266's fixture scope: decode output of a canonical input is itself a
   serializer fixed point (statement order matches the serializer's
   root-first post-order DFS exactly), including v0.5 state/query programs
   and dead-binding chains.
2. **`models/dsl_tokenizer.py` (`DSLNativeTokenizer`)** — aligned.
   `canonicalize(x_c)` of a serializer-canonical input is a serializer fixed
   point: `_pretty_print` spacing, double-quote style (`json.dumps`), and
   statement order all agree with the official serializer. The only collapse
   is deterministic alpha-renaming (`root`, `b1…`, `$s0…`), which is mirrored
   in the loss space (binders/states encode to `<BIND_j>`/`<STATE_i>` slots,
   so training loss is invariant to surface names) and preserved by the
   serializer. Id streams are fixed points of decode; encode→decode preserves
   the lang-core resolved structure modulo the alpha map.
3. **`models/tokenizer.py` (`OpenUITokenizer`)** — an exact identity codec
   (whitespace-preserving); it applies no collapse of its own, so canonical
   alignment is entirely delegated to the record seam.
4. **The training-target seam is the load-bearing invariant.**
   `TwoTowerModel._encode_openui` encodes `record.openui` verbatim for both
   output tokenizers; canonical space is guaranteed because
   `data.contract.normalize_example_record` (applied by the train_data
   pipeline to every document record) rewrites `openui` to
   `validate(x).serialized`. Verified: normalization output is a serializer
   fixed point and idempotent for all 36 seeds — including the 6 raw seeds
   whose statement order diverges from the serializer.

## Regression tests

`tests/test_dsl/test_canonical_alignment.py` (new, 9 tests, bridge-gated with
the existing `bridge_available()` skip pattern): serializer-fixed-point of
decode, loss-space idempotence, and alpha-modulo canonical equality for the
production codec and `DSLNativeTokenizer` over fixtures + targeted programs;
seam tests asserting `normalize_example_record` output is a canonical fixed
point (and that the raw seeds genuinely need the collapse), with both output
tokenizers' target encodings fixed points of their decode.

Suites: `tests/test_dsl` + `tests/test_models` green (counts in PR run log),
plus `scripts.repo_policy` and `ruff` on changed files.

## Honest caveats

- **Bridge-dependence:** without the Node bridge, `validate(x).serialized`
  is the Lark backend's input echo (`lark_backend.py` sets
  `serialized=source.strip()`), so `normalize_example_record` silently
  degrades to a non-canonicalizing pass-through. Corpora built without the
  bridge are not guaranteed canonical; the new tests skip rather than fail in
  that environment. Canonical-eval ownership of this asymmetry remains with
  D2 (SLM-30, `iter-e252-canonicalizer-20260716.md`).
- `scope_identity_*` records intentionally train verbatim echo (including
  de-canonicalized variants) with `preserve_verbatim`; the paired
  `scope_canonical_*` family supplies the collapse direction. These are
  excluded from the canonical-target claim by design, not overlooked.
- Fixture/corpus wiring evidence only — no checkpoint, matrix row, or ship
  claim.

## Conclusion

The reopened B2 scope closes with an audit result of **already aligned**: all
three codec paths compute loss in a space consistent with the official
lang-core canonical form, and the alignment is now pinned by bridge-gated
round-trip property tests rather than assumed.
