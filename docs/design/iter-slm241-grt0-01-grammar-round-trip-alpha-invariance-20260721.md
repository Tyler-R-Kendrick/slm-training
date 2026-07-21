# SLM-241 (GRT0-01): D2 canonicalizer round-trip / alpha-invariance stress probe (slm241-grammar-round-trip-alpha-invariance-20260721)

**Matrix set:** `slm241_grammar_round_trip_alpha_invariance`
**Version:** `grt0-01-v1`
**Status:** fixture
**Claim class:** wiring
**Gate hash:** `a9615c016fbf0bb4...`
**Disposition:** ceiling_confirmed_at_scale — All 150 generated candidates (149 with 2+ non-root binders) were idempotent under canonicalize, always re-validated, and were alpha-invariant under a full non-root binder permutation (0 permuted variants failed to even parse, which would itself be a renamer artifact, not counted against the claim). The canonicalizer's documented normal-form property holds across this generator-scale corpus, beyond the 3 hand-picked unit-test examples.

## Hypothesis

The D2 canonicalizer (slm_training.dsl.canonicalize.canonicalize, built on production_codec.encode_openui/decode_productions) is a stable normal form across a broad, coverage-guided corpus of generated OpenUI programs -- not just the 3 hand-picked examples in tests/test_dsl/test_canonicalize.py: (1) canonicalize is idempotent, (2) canonicalize's output always re-validates through the real parser, and (3) renaming every non-root local binder identifier to a fresh, disjoint set of names never changes the canonical form (alpha-invariance).

## Falsifier

Any generated candidate for which canonicalize(canonicalize(x)) != canonicalize(x); or canonicalize(x) fails to re-validate through slm_training.dsl.parser.validate; or a binder-permuted variant of x (same layout, only non-root local identifiers renamed) is grammar-valid but canonicalizes to a different string than x.

## Honest caveats

- Fixture/wiring evidence only: no checkpoint, GPU run, or ship-gate claim is made or implied.
- This is a positive/ceiling-style probe: it asks whether an existing, documented normal-form claim holds at generator scale, not whether some new mechanism should ship. It does not change canonicalize, production_codec, the parser, or any generator default.
- The corpus generator (ProgramGenerator._choose / _build_program) is the same coverage-guided candidate machinery generate_one() uses, but this harness calls it directly and validates candidates through slm_training.dsl.parser.validate (hybrid, Lark-fallback) instead of generate_one()'s verify_record path, because verify_record's G2 gate calls the bridge-only slm_training.dsl.lang_core.validate, which raises unconditionally when the official @openuidev/lang-core Node bridge is unavailable -- as it is in this sandbox. Cross-backend parity between the official lang-core parser and the in-process Lark grammar is therefore untested here; only the Lark-backed path is exercised.
- The binder renamer is a regex-based identifier substitution (string literals are masked first, then whole-word non-root binder identifiers are substituted via a single alternation regex, then literals are restored) -- not an AST-level rename. It is exercised against, and passes, the same placeholder-aliasing edge case already regression-tested in tests/test_dsl/test_canonicalize.py (binder stems appearing inside quoted placeholder text), but it is a test-harness utility, not a reusable production renamer.
- Candidates with 0 or 1 non-root binder (root refers to nothing else, or exactly one other statement) have no possible nontrivial permutation and are recorded as trivial (alpha_invariant=None), not counted toward the alpha-invariance claim.

## Recipe

- seeds: `[0, 1, 2]`, count per seed: `50`
- candidates scored: `150` (`149` with 2+ non-root binders, `1` trivial)
- mean `canonicalize()`+`canonicalize(canonicalize())` latency: `1.015 ms`
- backend: hybrid OpenUI (Lark fallback — the official lang-core Node bridge is unavailable in this sandbox)

## Per-seed summary

| seed | candidates | idempotent | revalidates | alpha-invariant (of non-trivial) | permuted-invalid |
| --- | --- | --- | --- | --- | --- |
| 0 | 50 | 50/50 | 50/50 | 50/50 | 0 |
| 1 | 50 | 50/50 | 50/50 | 49/49 | 0 |
| 2 | 50 | 50/50 | 50/50 | 50/50 | 0 |

## Counterexamples (if any)

None — every scored candidate was idempotent, re-validated, and alpha-invariant.

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `canonicalize`, `production_codec`, `slm_training.dsl.parser`, or any generator default, does not train a model, and makes no ship or gate claim. It scales the D2 canonicalizer's documented normal-form claim from 3 hand-picked unit-test examples to a generator-driven corpus and records the outcome honestly, whether confirming or falsifying.

## Reproducibility

```bash
python -m scripts.run_slm241_grammar_round_trip_alpha_invariance --mode plan-only
python -m scripts.run_slm241_grammar_round_trip_alpha_invariance --mode fixture
```
