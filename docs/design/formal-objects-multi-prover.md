# Exportable formal objects and multi-prover loop

**Status:** Implemented. Closes the formal loop for VSS `SupportCertificate`s
and Lean claim modules without relying on a single Lean kernel.

## Problem

Support certificates were replayable only inside the Python oracle path, and
Lean theorems lived only under the Lean 4 kernel. Either path alone is a
**single trust root**. A formal loop is closed only when **independent provers**
agree on exportable objects.

## Formal object

Schema: `formal_object/v1`  
Code: `src/slm_training/formal/objects.py`  
JSON Schema: `src/slm_training/resources/formal_object.schema.json`

| Field | Role |
| --- | --- |
| `kind` | `support_certificate` \| `lean_claim` \| `closure_law` |
| `statement` | Machine-readable claim (verdict, theorem id, laws) |
| `payload` | Certificate body or theorem metadata |
| `content_digest` | SHA-256 of kind+id+statement+payload |
| `required_checkers` | Minimum backends that must run |
| `tier` | `production_core` or `ecosystem_library` |

### Exports

| Source | Exporter |
| --- | --- |
| VSS `SupportCertificate` | `export_support_certificate` |
| Lean theorem catalog | `export_lean_claim` / `lean_claim_catalog` |
| Exact-closure honesty laws | `export_closure_law` |

## Independent checkers (provers)

| Backend | What it trusts | Used for |
| --- | --- | --- |
| `python_structural` | Pure list-based law encoding + certificate honesty | all kinds |
| `python_reference` | Independent set/digest-first re-encoding of the same laws | all kinds |
| `python_replay` | Full enumerative search replay (`replay_support_certificate`) when expander+verifier context is supplied | support certificates |
| `lean_kernel` | Optional `lake build LeverProofLean` | lean claims / closure laws |

**Rule:** Lean is never sole authority. `loop_requires_multi_backend` rejects
acceptance when only `lean_kernel` succeeds (`single-kernel reliance`).

Default `min_backends = 2`. Support certificates and Lean claims require
`python_structural` **and** `python_reference` by construction.

## Close the loop

```bash
# Catalog Lean claims + VSS closure laws (pure multi-prover, no Lean required)
python -m scripts.verify_formal_objects \
  --catalog-lean --closure-laws \
  --export-dir outputs/formal_objects \
  --out docs/design/formal-loop-report.json

# Optional third backend
python -m scripts.verify_formal_objects --catalog-lean --lean

# Support certificate file(s)
python -m scripts.verify_formal_objects --certificate path/to/cert.json
```

Report schema: `formal_loop_report/v1`  
Code: `src/slm_training/formal/loop.py`

A report has `closed: true` only when every object is accepted by enough
independent backends.

## Trust boundary

| Trusted | Not proved |
| --- | --- |
| Multi-backend agreement on the exported object | That measurements were unbiased |
| Certificate honesty laws + optional full replay | That the expander matches production OpenUI pack semantics without a provided context |
| Structural mirrors of Lean laws | That Lean source text bit-identically matches Python (module build is optional third) |

## Relationship to other formal surfaces

- Lean theories: `src/leverproof_lean/` ([core-formal-claims.md](core-formal-claims.md))
- Metric oracle certificates: LeverProof `metric_certificate/v*` (promotion bands)
- Ecosystem tier split: [ecosystem-tier.md](ecosystem-tier.md)
- VSS contract: [verified-scope-solver.md](verified-scope-solver.md)

## Honesty

This is a **formal-loop / infrastructure** claim, not a ship-quality claim. It
does not replace `--ship-gates`, training, or held-out evals.
