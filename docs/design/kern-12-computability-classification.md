# KERN-12 — Practical computability classification vocabulary (SLM-532)

Production-facing computability labels with **testable evidence contracts**,
separate from genuine reverse-mathematics interpretation claims (HARN-08 /
HARN-05). Extends KERN-10 / EVID-03 parity labels — not a parallel classifier.

## Artifacts

| Path | Role |
| --- | --- |
| `src/slm_training/formal/computability_classification.py` | Canonical vocabulary, contracts, validation |
| `src/slm_training/resources/formal/computability_classification.schema.json` | Claim / fixture schema |
| `src/slm_training/resources/formal/computability_classification_fixtures.v1.json` | Examples + counterexamples |
| `src/leverproof_lean/LeverProofLean/ComputabilityClassification.lean` | Lean inductive + label `#guard`s |
| `tests/test_formal/test_computability_classification.py` | CI contracts + fixture parity |
| `scripts/verify_computability_classification.py` | Standalone CI entrypoint |

Schema id: `computability_classification/v1`.

## Vocabulary

| Class | Meaning (short) | Required evidence (any one) | Does **not** imply |
| --- | --- | --- | --- |
| `finite_decidable` | Decision over a proved-complete finite domain (finite VSS) | `finite_vss_certificate`, `finite_search_bound` | `RCA0`, unbounded `total_recursive`, `rm_research:*` |
| `primitive_recursive` | PR / bounded-primitive computation with PR-shaped bound | `primitive_recursive_bound` | finite decidability without a finite domain; `RCA0` |
| `bounded_search` | Search over an explicit finite bound (prefix-tree / closure) | `finite_search_bound`, `bounded_closure_certificate` | full `finite_decidable` without domain completeness; `semidecidable` |
| `total_recursive` | Total recursive function with a totality certificate | `total_recursive_certificate` | `finite_decidable`, `primitive_recursive`, `RCA0` |
| `semidecidable` | RE / Σ₁ — halts on yes | `semidecision_procedure` | `co_semidecidable`, decidability |
| `co_semidecidable` | co-RE / Π₁ — halts on no | `co_semidecision_procedure` | `semidecidable`, decidability |
| `oracle_relative` | Relative to a named external oracle / black box | `oracle_declaration` | absolute computability |
| `classical_propositional` | Classical LEM / DNE without constructive witness | `classical_propositional_marker` | computable existence |
| `noncomputable_existence` | Existence without a computable witness | `noncomputable_existence_claim` | `RCA0` / `ACA0` via axioms alone |
| `unclassified` | Explicit refusal (incomplete domain / missing evidence) | `explicit_unclassified` | any positive class |

Legacy alias: `classical_noncomputable_existence` → `noncomputable_existence`
(KERN-10 / EVID-03 continuity).

### Research labels

`rm_research:<Subsystem>` is admitted **only** with an explicit interpretation
package (`base_theory_id`, `coding_id`,
`interpretation_status ∈ {explicit_interpretation, explicit_reversal}`,
`forward_evidence_sha256`, and `evidence_kinds` containing
`interpretation_package`). `#print axioms`, Lean compile success, finite
bounded analogues, and classical existence **cannot** alone yield
`rm_research:RCA0` / `WKL0` / … (same anti-overclaim law as HARN-08).

## Semantics notes

- **Decidable vs semidecidable:** finite decidability is a total decision on a
  complete finite domain. Semidecidability only guarantees halt-on-yes; it does
  not imply co-semidecidability or decidability.
- **Oracle-relative:** black-box / external oracle bounds are relative; they do
  not license absolute RE or recursive claims.
- **No ambiguous fall-through:** unknown class ids and empty evidence for a
  positive class fail closed.

## Fixture coverage

Positive: finite VSS, bounded closure, PR bound, oracle, semidecidable /
co-semidecidable, classical propositional, noncomputable existence (incl.
legacy alias), total recursive, unclassified, `rm_research:WKL0` with package.

Negative: `#print axioms` → `rm_research:RCA0`; `rm_research` without package;
`finite_decidable` with empty evidence; unknown class; `#print axioms` as
`finite_decidable`.

## Lean / Python parity

- Python `PRACTICAL_CLASSES` ↔ Lean `PracticalClass` constructors / `.label`
- Fixture `lean_label` strings match Lean `#guard` constants
- Both sides reject bare `rm_research:*` without a package (Lean via
  `rejectsRmResearchPrefix` + Python validation)

## Run

```bash
PYTHONPATH=src uv run pytest tests/test_formal/test_computability_classification.py -q
PYTHONPATH=src uv run python -m scripts.verify_computability_classification
(cd src/leverproof_lean && lake build LeverProofLean.ComputabilityClassification)
```

## Cross-links

- HARN-08 labeling: [reverse-mathematics-computability.md](reverse-mathematics-computability.md)
- KERN-10 parity: [kern-10-resource-bound-trigger-parity.md](kern-10-resource-bound-trigger-parity.md)
