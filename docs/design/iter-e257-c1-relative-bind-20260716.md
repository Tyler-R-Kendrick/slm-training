# E257 — C1 scope-as-relative-index binder references (2026-07-16)

Fixture-grade wiring row for Track C1. Machine-readable evidence:
[quality-matrix-results-iter-v10-c1-20260716.json](quality-matrix-results-iter-v10-c1-20260716.json)
(control E255 in
[quality-matrix-results-iter-v10-b4-20260716.json](quality-matrix-results-iter-v10-b4-20260716.json)).
Code: [`src/slm_training/models/dsl_tokenizer.py`](../../src/slm_training/models/dsl_tokenizer.py)
(`bind_encoding="relative"`). Linear SLM-25.

## What was built

A De Bruijn-style relative binder channel on the lexer-native tokenizer
(motivated by the neural binding problem, Greff et al. 2020 — externalize
binding to the verifier the same way syntax was externalized):

- **Definition sites are nameless**: `hero = Card([...])` emits `<BINDDEF>` —
  statement position *is* the identity, so the model never chooses a binder
  name at all (previously it picked an identity slot `<BIND_j>`).
- **Reference sites carry a signed statement delta**: `<BINDREL_+3>` = "the
  binder defined three statements after this one". OpenUI allows forward
  references (`root` typically references binders defined later), so the
  encoding uses signed deltas rather than classic most-recent-binder De Bruijn
  indices.
- **Scope legality is the verifier's job, not the model's**: decode inverts
  deltas positionally; an out-of-scope offset decodes to a never-defined
  `oob<k>` name that the stream verifier rejects (`stream_check` reports it
  unresolved) — never silently repaired. The grammar gate accepts the new
  tokens wherever `NAME` is legal (they carry `TokenKind.BIND`, which the
  fastpath token map expands for the `NAME` terminal).
- Encoding is persisted in the tokenizer sidecar (`bind_encoding`), plumbed
  through `TwoTowerConfig` / `ModelBuildConfig` / matrix `Experiment`
  (default `absolute` — zero behavior change for every existing row), and
  requires root-first canonical statement order (raises otherwise).

## Canonicalizer alpha-invariance fix (D2 interaction)

The C1 round-trip property test exposed a real bug in the D2 canonicalizer's
statement ordering (`dsl/production_codec.py::_statement_order`): references
were extracted from statement RHS **including string-literal contents**, so
placeholder text like `":form.title"` aliased the binders `form`/`title` and
the canonical statement order (hence the whole canonical form) depended on the
original binder names — breaking the alpha-invariance the canonicalizer
claims. Fixed by stripping string literals before reference scanning;
regression test `test_alpha_invariance_when_placeholders_alias_binder_names`
in `tests/test_dsl/test_canonicalize.py`.

## Verification

- Round-trip property test over every committed seed program: encode(relative)
  → decode → `canonicalize(decoded) == canonicalize(original)` through the
  official lang-core bridge (`tests/test_harnesses/model_build/test_dsl_tokenizer_relative.py`).
- Unit tests: nameless definitions + expected deltas on a forward-reference
  program, root-first enforcement, out-of-scope rejection via `stream_check`,
  grammar-gate NAME coverage, sidecar persistence, matched-row registration
  (`tests/test_scripts/test_quality_matrix_v11.py`).

## Fixture result (wiring evidence only)

E257 vs the matched E255 control (identical recipe: `--scratch-control`,
200 steps, lr 3e-4, batch 4, seed 0, CPU, fixture v1 corpus 108 records,
parallel MaskGIT decode; suites smoke 3 / held_out 5 / adversarial 4 / ood 4 /
rico_held 0):

| Metric | E255 absolute | E257 relative |
| --- | --- | --- |
| syntax parse (sm/ho/adv/ood) | 0.0 / 0.0 / 0.0 / 0.0 | **0.667 / 0.6 / 0.25 / 0.5** |
| meaningful parse | 0.0 everywhere | 0.0 everywhere |
| structural similarity | 0.30 / 0.32 / 0.28 / 0.37 | 0.27 / 0.32 / 0.41 / 0.21 |
| train loss @200 | 3.75 | 3.27 |
| decode p50/record | ~15s | 1–8s |

At fixture scale, removing binder-identity decisions moves the model from
zero syntactically valid outputs to a 0.25–0.67 syntax-parse rate under the
same honest gates, with lower loss and much cheaper decode. **Meaningful parse
stays 0.0 on both rows** — the failures shift from placeholder-literal
violations to `empty_root_stack`/underfull layouts, i.e. C1 does not touch the
valid-but-empty wall (that is Track A's target, consistent with the program's
division of labor).

## Honesty and limits

- Fixture/scratch wiring evidence only: 108-record corpus, tiny suites, no
  ship claim, no gate weakened, nothing promoted. The E-row at frontier scale
  (real corpus, GPU, full suites) remains open, as does the C1×A interaction
  (does relative binding compound with the A2–A4 emptiness fixes?).
- The compiler-tree decode legality layer (`compiler_draft._binder_scope`,
  V9) currently reasons over absolute `<BIND_j>` declarations and is not yet
  relative-aware; E257 therefore runs parallel MaskGIT decode. Making the V9
  lattice rows relative-compatible is follow-up work.
- OpenUI 0.2.x has little binding surface; the encoding is designed
  forward-compatible with richer binding DSLs (GraphQL, F2) where the
  relative channel should matter more.
