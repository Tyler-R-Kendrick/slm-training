# DSH2-07 Two-pack generic CAP1 freeze + schema-sensitivity suite (SLM-367)

**Decision:** supported at the contract-fixture evidence level. A generic
grammar/schema-grounding claim no longer has to rest on OpenUI alone:
`src/slm_training/dsl/mini_pack.py` ships **mini-flow**, a complete test-only
DSL pack with zero OpenUI imports, and
`src/slm_training/harnesses/train_data/cap1_two_pack_freeze.py` freezes an
immutable two-pack CAP1 suite (`Cap1TwoPackSuiteV1`) over **disjoint semantic
root families** with sha256-addressed rows, a tamper-evident manifest, and
suite cards carrying exact generation/adjudication provenance.

Machine-readable evidence:
[iter-slm367-two-pack-freeze-20260725.json](iter-slm367-two-pack-freeze-20260725.json)

## The second pack (mini-flow)

Grammar: `src/slm_training/dsl/grammars/mini_flow.lark` — a task-flow language
(`let` bindings, `task name { step "..." mode fast|safe retry N }`) with one
closed value domain (`mode`), a placeholder-bearing content prop (`step`), and
an open integer prop (`retry`). The pack bundles every slot the generic claim
needs:

- Lark grammar authority (DSH1-01 `lark_authority`, start symbols
  `start` + `prop_stmt` — the fragment surface);
- deterministic canonicalizer (idempotent codec round-trip);
- static validator (parse-or-raise; closed domain enforced by the grammar);
- schema contract (`CAP1SchemaV1`, constructed directly — the dataclass is
  DSL-agnostic);
- `SemanticFrameV1` provider (DSH2-02) emitting required closed-value facts
  plus forbidden excluded-value complements, placeholder content facts, and
  let-bindings;
- marker (placeholder) policy; oracle labeled `syntax_only` (F3 honesty).

**Stop rule (fail-closed):** `require_complete_for_generic_claim` runs a
DSH1-01-style conformance check — every required capability must be present
*and* pass a smoke probe. An incomplete pack raises `PackIncompleteError` and
cannot support a generic claim (tested: missing authority, missing frame
provider, missing schema contract all fail closed).

**Generic-code audit:** tests statically parse `mini_pack.py` and forbid any
`openui` import, audit the grammar surface, and derive frames with the OpenUI
pack registry unavailable. Mini rows in the suite carry `dsl: "mini-flow"`
schema contracts.

## Strata

Per base case (two per pack): `original`, `schema_empty`,
`schema_contradictory` (used closed values *and* used components removed),
`schema_reordered` (key order only), `schema_irrelevant` (unrelated
extension), `marker_permutation` (SLM-366 `marker_permutation` authority),
`paraphrase` (SLM-365 offline fixture provider for OpenUI; deterministic
mini renderer), `counterfactual` (SLM-366 one-fact pair for OpenUI;
DSH2-02 `verify_single_fact_change` flip for mini-flow), `ambiguity`, and
`cap0_retention` (identity round-trip).

**Ambiguity:** no SLM-344 accepted-set evidence exists in-repo, so ambiguous
rows are either scored against a recorded local fixture accepted set
(`disposition="accepted_set"`, ≥2 targets) or explicitly excluded with a
recorded reason (`disposition="excluded"`); excluded rows never enter
determinacy scoring.

## Scoring + gates

`score_schema_sensitivity(suite, decide_fn, tolerance=0.0)` — relevant
perturbations (empty/contradictory schema, counterfactual) **must** change
the intended decision; reordered/irrelevant perturbations must keep it
invariant within `tolerance`. Tested with a synthetic schema-grounded
`decide_fn` (sensitivity 1.0 / invariance 1.0), a jittery decider (tolerance
absorbs one flip), and a constant decider (fails).

`adjudicate_suite` additionally scores canonical AST/equivalence,
required/forbidden fact satisfaction, prompt invariance, determinacy
calibration, and CAP0 retention. `evaluate_gates` makes **CAP0 retention
mandatory**: a suite without retention rows per pack fails closed (tested),
as does a decider that lost the identity behavior.

## Fixture results

| Metric | Value |
| --- | --- |
| Rows | 41 (2 packs × 2 base cases × 10 strata + 1 excluded ambiguity row) |
| Relevant sensitivity | 1.0 |
| Invariant rate | 1.0 |
| Facts accuracy | 1.0 |
| Prompt invariance | 1.0 |
| CAP0 retention | 4/4 rows, accuracy 1.0, both packs |
| Determinacy | 36 determinate single-target rows, 4 accepted-set, 1 excluded (reason recorded), calibrated |
| Gates | passed |
| Integrity | fail-closed `verify_suite_integrity` ok; tamper + manifest-swap detected in tests |

**Freeze:** `freeze_suite` writes `cap1_two_pack_suite.json` create-once
(claim-manifest pattern); divergent rewrites raise `FileExistsError`,
integrity-violating suites are refused.

Tests: `tests/test_dsl/test_mini_pack.py` (12) +
`tests/test_harnesses/train_data/test_cap1_two_pack_freeze.py` (23) — 35
passed, deterministic suite hash across rebuilds.
