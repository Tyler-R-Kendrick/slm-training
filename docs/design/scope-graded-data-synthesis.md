# Scope-graded data synthesis: identity anchors, canonical bias, scoped repair, typed lexical maps

Status: **deterministic corpus built and verified; no training run, no ship claim**.
Machine-readable evidence: [scope-graded-data-synthesis-results.json](scope-graded-data-synthesis-results.json).

## Problem

The corpus optimized for the wrong conditions (see
[local-decision-interventions.md](local-decision-interventions.md)): supervision
conditioned on gold states and prose-intent prompts, with no data that (a)
anchors the model on its own DSL surface, (b) biases decoding toward the
canonical tree form, or (c) repairs realistic typos below document scope.

## Design

Four deterministic families, each emitted at every AST-derived lexical scope
(document / statement / expression / lexical). Scope extraction is
grammar-generic: `slm_training/data/scope_extract` walks a position-preserving
Lark parse tree obtained from the `GrammarBackend` layer (`LarkFileBackend.parse_tree`,
falling back to `info.grammar_path`), so any registered grammar — proven with
`toy-layout` — yields the same slices. Nothing is keyed to OpenUI component
names.

| Family | Input → target | Task | Purpose |
| --- | --- | --- | --- |
| `scope_identity_{scope}` | source slice → byte-identical slice | `identity` | intentional memorization of the DSL's own surface |
| `scope_canonical_{scope}` | non-canonical variant slice → canonical slice | `edit` | bias toward the tree-optimized (canonical) form |
| `scope_repair_{scope}` | verified typo/mistake → corrected slice | `repair` | sub-document repair (document stays `corruption_repair`) |
| `lexical_typed_map` | surface token → typed node, e.g. `true` → `Boolean(true)` | `generation` | token → typed-AST mapping (`typed_node` output kind) |

Producer: `harnesses/train_data/scope_corpus.py`, wired into
`build_train_data` from ProgramSpec roots (same lineage / split groups, so
split isolation holds). Key mechanics:

- **Identity integrity.** Sub-document rows already pass `_normalize_record`
  verbatim; document rows carry `meta.preserve_verbatim`, which validates the
  program but skips style-strip/re-serialization and stamps
  `serialization: preserved_verbatim`.
- **Canonical pairs.** Deterministic de-canonicalization variants
  (`rotate_statements`, `tighten_whitespace`, `single_quote_strings`) are
  fail-closed: each must parse and round-trip (`validate(variant).serialized ==
  canonical`). Slices align across the rewrite by statement anchor +
  intra-statement AST path; unmatched slices drop. Every canonical row has an
  identity twin with the *same prompt* — the ranking bias decides between the
  verbatim echo and the canonical rewrite.
- **Ranking bias, both ways.** `mixture.default_base_weights` gives
  `scope_canonical_*` (0.03) more weight than `scope_identity_*` (0.02), and
  each build writes `preference_pairs.jsonl` (chosen = canonical, rejected =
  verbatim, `pair_corpus: canonical_bias`) for the preference harness. A new
  `identity_echo` task group lets mixture/diffusion training dial the
  memorization anchors independently; per-scope families do the same for
  scopes.
- **Scoped repair.** `data/corrupt/oracle.build_scoped_corruptions` reuses the
  corruption operators (plus deterministic lexical typos: transposed chars,
  dropped closing quote) with `validate_output` as the fail-closed rejector —
  clean must pass, broken must fail, per fragment kind.
- **Typed maps.** `typed_render` derives `Boolean(true)` / `Number(42)` /
  `String("…")` from grammar terminal names and the shared typed-terminal
  conversion; a `typed_node` output kind validates the constructor shape.
  (A grammar fix rode along: `BOOL.2` terminal priority, so `true` lexes as a
  boolean instead of a `NAME` ref — matching lang-core semantics.)
- **Exposure honesty.** With scope families in the build, `apply_parent_cap`
  caps per (family, parent) instead of per parent, so the multiplied per-root
  rows stay bounded per family without evicting each other; `family_stats` and
  `synthesis_telemetry.jsonl` account per family. The independent judge gained
  an `identity_echo_mismatch` check and treats scope prompts (which embed DSL
  source, not prose intent) as non-semantic requests.

## Measured results (2026-07-16, CPU, pinned bridge)

Recipe: `python -m scripts.build_train_data --source all --version
scope_graded_v1 --output-root outputs/data/train`, built twice.

| Check | Result |
| --- | --- |
| Reproducibility | both builds `56db51d38947b090fb901c740e7035549743437e935b0047ea124997544d623a` |
| Accepted rows | 1,223 total; 466 scope-graded across all 12 families |
| Scope groups | identity 227 · canonical 144 · repair 68 · typed 27 |
| Preference pairs | 189 (all chosen ≠ rejected, chosen ranked higher) |
| `verify_data_synthesis` `scope_corpus` check | pass — identity echo, canonical round-trip, typed-render, and pair invariants all hold |
| Tests | 326 passed / 2 skipped across `tests/test_data`, `tests/test_harnesses/train_data`, `tests/test_dsl`; new suites `test_scope_extract`, `test_scope_corpus`, `test_output_kinds` |
| Errors | 0 build errors |

Honesty caveats: data-only campaign (no checkpoint, no ship claim);
document-scope rows for roots that fail the document quality bar are rejected
by the same gates as any other document row (their sub-scopes still pass); the
`single_quote_strings` variant serves as input only — the layered verifier
quarantines it as a target, so its document identity twin is skipped
fail-closed.

## Follow-ups

- Register an experiment-matrix row training on `scope_graded_v1` with an
  overfit probe (high steps, small root set) to measure verbatim echo
  fidelity, and a diffusion run isolating scopes via family/task weights.
- Extend roots beyond ProgramSpecs (fixture/rico) if scope coverage of organic
  layouts is needed.
