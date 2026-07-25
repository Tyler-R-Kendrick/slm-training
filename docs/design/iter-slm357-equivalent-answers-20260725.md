# DSH1-05 expanded-equivalent answers + canonical preference (SLM-357)

**Decision:** supported at the deterministic contract-fixture level.
Nonminimal but semantically equivalent program forms are registered and
materialized **without** contradictory SFT targets or shortest-valid
collapse: semantic equivalence and canonical preference are separate
relations, and canonical cost is compared only inside a compiler-verified
equivalence class.

Machine-readable evidence:
[`iter-slm357-equivalent-answers-20260725.json`](iter-slm357-equivalent-answers-20260725.json).

## Transform registry with proof obligations

`equivalence_transform_registry/v1`
(`src/slm_training/harnesses/train_data/equivalence_transforms.py`)
registers pack-approved `EquivalenceTransformV1` entries across four kinds —
explicit defaults (style-literal default args the canonicalizer strips),
alias (binder alpha-renaming), redundant grouping (top-level statement
rotation), and noncanonical formatting (separator whitespace). Every
transform carries preconditions naming its semantic authority, an `apply`
body, and provenance. `prove_equivalence` admits an application only when
the expanded form parses through the official parser **and** canonicalizes
byte-identically to the canonical input (same canonical AST / equivalence
class). Transforms that cannot prove equivalence on a fixture are removed or
retained only as negative fixtures — the stop rule, exercised by the
registered `swap_container_children` transform (order-sensitive, no
authority), which is provably rejected and never becomes an accepted output.

## Relations and canonicality ordering

Materialization emits `EQUIVALENT_TO`, `CANONICALIZES_TO`, and
`CANONICALLY_PREFERRED_TO` relation records with deterministic per-kind
margins (`KIND_MARGINS`: explicit_defaults 0.05, alias 0.10,
redundant_grouping 0.15, noncanonical_formatting 0.20), transform versions,
and provenance stamps. `compare_canonicality` orders by canonical cost
(surface tokens, then chars) **only** when both candidates satisfy every
required role with equal cardinality, parse under the same scope, and share
one canonical fingerprint; different classes are unordered. A candidate
missing a required role is semantically incomplete and can never outrank a
complete one, no matter how short (tested with a shorter-but-incomplete
program missing `:cta`).

## Materialization

`materialize_sft_records` emits the canonical form as the primary SFT target
(`ExampleRecord.openui`), proven expansions as `accepted_outputs`, and
chosen=canonical / rejected=expanded `PreferencePair`s with deterministic
scores (1.0 vs 1.0 − margin). When the task explicitly asks for expansion
(`prefer_expanded=True`), the expanded form becomes the primary target and
chosen output, and the canonical form moves to `accepted_outputs` /
rejected — tested. Order-sensitive properties (container child order) are
never reordered without authority.

## Results

- `tests/test_harnesses/train_data/test_equivalence_transforms.py`: 9 passed
  (same-class canonicalization, proof obligation per registered transform,
  stop rule, order-sensitive guard, incomplete-never-outranks, within-class
  ordering, canonical-primary pairs, prefer-expanded flag,
  determinism/provenance).
- Train-data + canonicalizer suites: same pass/fail profile as the SLM-356
  baseline on this branch (5 pre-existing failures: 4x
  `test_staged_materialization.py`, plus the dsh0-02 symbolic-surface
  evidence hash drift); this change adds no new failures.
- `python -m scripts.verify_version_stamps --check`, `repo_policy`, ruff,
  `git diff --check`: passed.

Registry: new component `harness.experiments.slm357_equivalent_answers` v1;
`harness.train_data` bumped to v21 (new module under its watched directory).

Claim limits: fixture-scale contract evidence only — no corpus publication,
no model evaluation, no ship-gate claim. Equivalence is proven per
application via parser + canonicalizer round-trip; unproven transforms are
negative fixtures, never accepted outputs.
