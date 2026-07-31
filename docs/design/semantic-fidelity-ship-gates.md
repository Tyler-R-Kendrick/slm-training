# Semantic-fidelity ship gates (BEq analogues)

**Status:** Implemented as `openui_ship_gates_v6`. Promotion cannot rest on
syntax parse or soft structural similarity alone.

## Problem

Soft metrics (`structural_similarity`, placeholder overlap) can stay green
while the model emits a different AST/grammar program. Certificate-backed
solver paths need digest-level agreement, not a bare verdict string. Lean-side
`BEq` on formal objects is the reference rigor; ship gates now carry the same
class of check on the production scoreboard.

## Predicates (Boolean equality)

| Predicate | Code | Meaning |
| --- | --- | --- |
| `ast_beq` | `evals/semantic_fidelity.py` | Structure-normalized exact match after validate/serialize (style-stripped). Near parity to AST surface equality. |
| `canonical_beq` | same | `canonical_equal` — full D2-canonical layout equality. |
| `certificate_equivalent` | same | SHA-256 digests of two support certificates / formal objects agree. |

Rates:

- `ast_beq_rate` — mean of `ast_beq` over document rows (also aliased from historical `exact_match` in slim projection).
- `canonical_beq_rate` — mean of `canonical_beq`.
- `certificate_equivalence_rate` — mean of digest BEq over certificate pairs when present.

## Ship policy (`DEFAULT_SHIP_GATES`)

Every policy suite now floors **both** BEq rates in addition to density and
structure. Floors are below `structural_similarity` (exact equality is harder)
but **strictly positive**, so a syntax-only or soft-structure-only run fails.

| Suite | `ast_beq_rate` | `canonical_beq_rate` |
| --- | ---: | ---: |
| smoke | 0.20 | 0.10 |
| held_out | 0.15 | 0.08 |
| adversarial / ood | 0.08 | 0.04 |
| rico_held | 0.05 | 0.02 |

### Certificate integrity (conditional)

When a suite reports `certificates_compared > 0` or `certificates_emitted > 0`:

- `certificate_equivalence_rate >= 1.0` (digest BEq)
- `certificate_replay_failures == 0`

Suites without certificates are unaffected (no vacuous green).

## What still does not promote

- `parse_rate` / `syntax_parse_rate` alone
- Soft `structural_similarity` without BEq floors
- DESIGN.md style lint
- Fixture-scale `n` below the suite floor

## Wiring

| Surface | Path |
| --- | --- |
| Predicates | `src/slm_training/evals/semantic_fidelity.py` |
| Scoreboard | `eval_runner` emits rates + per-row flags |
| Gates | `ship_gates.py` (`openui_ship_gates_v6`) |
| Formal export | certificate digests already export via `formal_object/v1` |

## Honesty

This tightens ship gates (never weakens). Models that previously cleared soft
structure floors without exact AST agreement will now fail until fidelity
improves. That is intentional.

## Related

- [adversarial-review.md](adversarial-review.md) — honest ship policy
- [structure-only-eval.md](structure-only-eval.md)
- [formal-objects-multi-prover.md](formal-objects-multi-prover.md) — certificate digests
- [core-formal-claims.md](core-formal-claims.md)
