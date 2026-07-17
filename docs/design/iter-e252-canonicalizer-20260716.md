# E252 — D2 OpenUI AST canonicalizer + canonical exact-match (2026-07-16)

Library + eval primitive, not a train/ship run. Code:
[`dsl/canonicalize.py`](../../src/slm_training/dsl/canonicalize.py),
[`evals/canonical_match.py`](../../src/slm_training/evals/canonical_match.py).
Linear SLM-30.

## What and why

Two OpenUI programs that differ only in binder names, statement order, or style
literals denote the same layout. Without canonicalization, surface exact-match
under-counts correct predictions and repeated-subterm detection (C3 macro
induction) misfires on surface form. D2 gives each equivalence class one
representative string.

## Mechanism (honest framing)

`canonicalize(source)` is a **confluent codec round-trip**: parse to the
grammar-native production stream (`production_codec.encode_openui`) and
deterministically re-emit (`decode_productions`). The codec already fixes one
canonical statement order (topological, first-use), renames binders to a De
Bruijn-style `v0, v1, …` pool, and strips style literals, so the round-trip is a
normal form by construction. The result is validated back through the official
parser and the transform is idempotent.

Verified: `root = Stack([hero],"column"); hero = Card([t]); t = TextContent(":x")`
and its binder-renamed twin both canonicalize to
`root = Stack([v1], "column"); v0 = TextContent(":x"); v1 = Card([v0])`.

**Not** an e-graph / equality-saturation engine and **not** a semantic simplifier:
it does not elide schema defaults or flatten containers (meaning-changing
rewrites, left to a future schema-checked pass). The name refers only to the
normal-form property.

**Caveat (arXiv:2401.02948).** The binder renaming is context-*insensitive*
De Bruijn-style — it canonicalizes alpha-equivalent whole programs but does not
by itself detect all context-sensitive common subterms. C3 must not assume
canonical binder identity implies shared subterm context.

## Eval consumer

`evals/canonical_match.canonical_exact_match_rate(pairs)` reports canonical vs
surface exact-match and a `canonicalization_rescued` count (correct layouts the
surface metric missed) — beside surface exact-match, never replacing it, never a
ship gate.

## Downstream

- D1 (simplification-consistent corruption) uses `canonicalize` as the forward-
  process simplifier.
- C3 (macro induction) uses `canonical_fingerprint` for repeated-span detection,
  subject to the caveat above.

## Verification

- `tests/test_dsl/test_canonicalize.py`: validates, idempotent, alpha-equivalent
  programs canonicalize equal, distinct layouts differ, unparseable → not equal.
- `tests/test_evals/test_canonical_match.py`: rescues an alpha-renamed correct
  prediction, counts true mismatches, agrees with surface on identity.
- 9 tests green. No checkpoint, no scoreboard, no ship claim.

## Correction (2026-07-16, C1 follow-up)

The C1 round-trip property test found an alpha-invariance violation:
`production_codec._statement_order` extracted references from statement RHS
**including string-literal contents**, so placeholder text aliasing binder
names (e.g. `":form.title"` vs binders `form`/`title`) made the canonical
statement order — and therefore the canonical form — depend on the original
binder names. Fixed by stripping string literals before reference scanning;
regression `test_alpha_invariance_when_placeholders_alias_binder_names` in
`tests/test_dsl/test_canonicalize.py`. Details:
[iter-e257-c1-relative-bind-20260716.md](iter-e257-c1-relative-bind-20260716.md).
