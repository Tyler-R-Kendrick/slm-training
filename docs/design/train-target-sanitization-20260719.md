# Deterministic train-target sanitization (AST optimization + templatization)

**Date:** 2026-07-19 · **Components:** `harness.train_data` v5, `data.test_build` v2
· **Results JSON:** [`train-target-sanitization-results.json`](train-target-sanitization-results.json)
(writer-emitted enforce-build `quality_report.json`, `version_stamp/v1`)

## What shipped

A deterministic sanitization pass over generated document targets, run inside
`_normalize_record` (`harnesses/train_data/pipeline.py`) **before** the
official `validate`, behind `sanitize_mode: off | audit | enforce`
(strict profile → `enforce`, permissive → `off`; CLI `--sanitize-mode`).
The same transform runs in the test-data builder
(`TestDataConfig.sanitize_mode`, default `enforce`) so future eval gold
matches the sanitized train distribution.

Per eligible target, in order:

1. **Schema-checked AST optimization** (`dsl/analysis/optimize.py`):
   D2 canonicalization (statement order, De Bruijn binders) plus the rewrites
   the D2 canonicalizer explicitly deferred —
   *elide trailing schema defaults* (curated `SCHEMA_DEFAULTS`),
   *drop dead bindings* (a G3-quarantine **rescue**), and
   *flatten single-child Stacks* (guarded; see below).
2. **Content-literal templatization** (`dsl/analysis/templatize.py`):
   literal strings → `:binder.slot` placeholders named purely by canonical AST
   position (namespace = canonical statement binder, slot = prop name,
   `_2, _3…` ordinals; alpha-equivalent inputs produce identical bytes).
   Original strings persist only as meta provenance
   (`meta["sanitize"]["template_fills"]`), never prompt/eval-visible.
3. **Official re-validation**: the bridge parser re-checks grammar, schema and
   the placeholder content policy; the stored form is the serializer output
   (a G8-compatible fixed point). Any failure at any step falls back to the
   unchanged input with a recorded reason — sanitize never drops a record.

## Load-bearing discoveries during implementation

- **The official parser rejects literal content props outright**
  (`content prop 'title' must be a placeholder…`). Literal-bearing candidates
  die at the normalize stage today, so templatization had to move **ahead of**
  `validate` (the plan originally placed it after). Templatization is
  therefore a *rescue*: it converts previously-unadmittable candidates into
  policy-valid ones. The regression suite proves the rescue end-to-end
  (`test_quality_report.py`, `test_sanitize.py`).
- **Templatization scope is the dual of the two literal gates**: every scalar
  string in a `CONTENT_PROPS` position (parser policy), plus free-form scalar
  strings in other props — exactly what `assess_record` hard-rejects as
  `non_placeholder_string` — with that gate's enum-like `[a-z0-9_-]+` /
  numeric exemptions mirrored, plus schema-enum, style-token and structural
  guards. Array strings are counted, not rewritten (v1).
- **`canonical_equal` certifies only the D2 half.** Default elision,
  flattening and templatization are outside D2's equivalence classes; a new
  `semantic_fingerprint` (render tree with `SCHEMA_DEFAULTS` filled, binder
  identity erased, dead statements invisible) certifies elide/dead rewrites at
  runtime. Flattening intentionally changes the fingerprint and is certified
  by schema child-admissibility (`children.items.anyOf` refs — e.g.
  `Card.children` is a restricted union), a root-top-node exclusion, a
  prompt-mention protection set (`_prompt_component_mentions`, so the judge's
  `prompt_component_missing_from_output` cannot fire), and parser
  re-validation. Statement-level flatten of a lone-ref wrapper redirects the
  reference sites and deletes the wrapper (no alias statements).
- **`SCHEMA_DEFAULTS` must be curated**: the schema snapshot has exactly one
  machine-readable default (`Form.fields: []`); Stack's
  `direction="column"` / `gap="m"` exist only in description prose. The table
  `{(Stack, direction): "column", (Stack, gap): "m", (Form, fields): []}` is
  test-pinned against `library_schema()` descriptions/enums so schema drift
  fails loudly.
- **Reserved-structure decontamination had to be augmented, not bypassed**:
  sanitized records live in a different structural-fingerprint family than
  raw fixtures. The build now reserves **both** each test fixture's raw and
  sanitized fingerprints (`sanitized_reserved_structures`), and the test-data
  builder leak-checks both the sanitized and pre-sanitize forms of every
  record — strictly more is rejected, never less.
- `transformation_lineage` is overwritten wholesale by
  `catalog.annotate_lineage`; sanitize provenance lives in a dedicated
  `meta["sanitize"]` block.

## Measured evidence (2026-07-19, `--source all --synthesizer none`, 921 seeds / 1073 collected)

| Metric | off | audit | enforce |
| --- | --- | --- | --- |
| `record_count` | 457 | 457 | **454** |
| content_fingerprint | `1834412598…` | `1834412598…` (**byte-identical to off**) | `5255b48cad…` |
| dedup drops | 559 | 559 | 566 (+7: canonicalization collapses alpha-variants) |
| reserved-structure drops | 8 | 8 | 4 (see analysis below) |
| n-gram eval flags | >0 (`eval_overlap_flagged`) | >0 | 0 |
| mean quality score | 0.9908 | 0.9908 | 0.9907 |
| placeholder vocab | 158 | 158 | 163 |
| judge pass rate | 1.0 | 1.0 | 1.0 |
| sanitize fallbacks | — | **0** | **0** |
| build wall time | 18.6 s | 17.2 s | 13.5 s |

Sanitization counters (identical audit vs enforce, as designed):
310 targets sanitized (140 unique / 186 cache hits), **337 defaults elided**,
**12 containers flattened** of 30 opportunities (18 blocked by guards),
0 dead bindings among current candidates (G3 already quarantines them
upstream; the rescue path is regression-tested), 0 literals templatized —
every committed document source is already placeholder-form, so the
templatizer's corpus effect awaits literal-bearing producers (the rescue is
proven by tests, not by this corpus). Skips: 636 non-document, 48
`preserve_verbatim`, 48 `scope_slice`, 15 `task_edit`, 16 `task_repair`.

Eval side (`build_test_data --source both` vs the matching train manifest):
enforce kept 42 records (14 leakage-rejected) vs off 43 (13) — the dual-form
leakage check caught one additional overlap; 42/42 sanitized, 40 defaults
elided, 0 fallbacks, 2.7 s.

**Decontamination precision analysis.** Under enforce, reserved-structure
drops move 8→4 and per-family `decontamination_drops` shift (rico_real 8→2,
design_md_contrastive 3→2, +2 new on human_curated). Both fingerprint families
are reserved, so this is not a weakening: raw-form candidates still match raw
fingerprints, sanitized candidates match sanitized fingerprints, and matching
becomes *canonical-vs-canonical* — surface-coincidental matches (binder
naming/order) stop firing while true isomorphs (including newly-rescued
trivial layouts on human_curated) are caught. Surface n-gram overlap with raw
eval suites likewise disappears under canonical renaming (0 flags); once test
builds run enforce (this change's default), the n-gram spaces re-align.
Dup-share on template families rises slightly (rico_real 0.79→0.83) — the
intended collapse of canonical near-twins, mirrored by the +7 dedup drops.
The pre-existing synthesis-feedback recommendations (awwwards quarantine
yield, rico redundant expansion) are unchanged by this slice and stay with
their producers.

## Decisions recorded

- **Flatten ships enabled** and **test builds default to enforce**
  (user-confirmed against the design agent's cautious alternatives). The
  eval-history discontinuity is stamped (`data.test_build` v2), never silent;
  committed snapshots are immutable and unaffected.
- **No new rejected.jsonl stage in v1**: transform failure always falls back
  to the unchanged input, which then faces every existing gate exactly as
  today. A strict-reject knob is a future option once fallback rates are
  observed in production builds (0 so far).
- **`integrity.py` untouched**: the SDE2-02 memo reserves `integrity_gate_mode`
  pipeline wiring as its own slice; a `SANITIZE_IDEMPOTENT` check there would
  collide. Idempotence is enforced inside `sanitize_openui` (fixpoint sweeps +
  re-emission stability + fallback) and asserted by tests; `evaluate_integrity`
  passes over sanitized records (regression-tested).
- **Preference pairs / scope corpus / identity anchors are fully skipped**;
  the canonical-bias pairs keep teaching official-serializer canonicalization.

## Follow-ups

1. **Eval-metric re-baseline**: train targets are now D2+optimized+templatized
   while committed eval gold predates the pass; `canonical_exact_match`
   rescues only the D2 half. Rebuild eval suites under `data.test_build` v2
   and re-baseline `_tree_match`/canonical metrics before comparing eval
   history across the bump.
2. Align preference-pair `chosen` targets with the sanitized distribution.
3. Array-string and richer non-scalar templatization (counted today:
   `skipped.array_string`).
4. Wire integrity gates into the pipeline per the SDE2-02 follow-up, including
   a sanitize-idempotence check.
5. Producer fixes for the pre-existing feedback recommendations (awwwards
   quarantine yield 0, rico dup-share) — unrelated to this slice.

No checkpoint was created or promoted; MODEL_CARD/README are unchanged.
