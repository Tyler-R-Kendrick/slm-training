# CAP0-03 (SLM-79): exact robust-code constructions and residual-precision reference functions

**Date:** 2026-07-18. **Status:** wiring / mathematical evidence only. This note
adds self-contained, Torch-free coding-theory and residual-precision reference
functions layered on the committed CAP0-02 arity package. It makes **no model,
train, eval, quantizer, checkpoint, or ship claim**. It records exact, replayable
facts about small finite constructions and integer predicates; it does not touch
decode or deployment behavior.

Owner package:
[`src/slm_training/dsl/analysis/arity/`](../../src/slm_training/dsl/analysis/arity/)
— specifically `coding.py`, `precision.py`, and `suggest.py`.

## Relationship to CAP0-02 (read first)

CAP0-02 ([`cap0-02-arity-analyzer-20260718.md`](cap0-02-arity-analyzer-20260718.md))
owns the canonical arity pipeline: `enumerate -> canonical trie -> acyclic
Myhill-Nerode minimisation -> K^d capacity`, exposed through `canonical.py`,
`state_graph.py`, `minimize.py`, and `report.py` (`ExactArityReport`,
`min_alphabet_for_capacity`). CAP0-03 does **not** re-derive any of those counts.
It consumes a target *state count* (the minimised-DFA class count CAP0-02
certifies) and answers robust-coding feasibility questions about it.

The only point of contact with the canonical API is
`coding.smallest_injective_arity`, which delegates the `K**d >= states` search to
CAP0-02's `report.min_alphabet_for_capacity` — exact integer arithmetic, one
source of truth, no float `pow`/`ceil` rounding.

## What was added

- `coding.py` — exact coding-theory reference functions and verified
  constructions:
  - `smallest_injective_arity` (delegates to `min_alphabet_for_capacity`),
    `hamming_ball_volume`, `hamming_sphere_packing_holds`,
    `gilbert_greedy_guarantees`, `singleton_upper_bound`.
  - `minimum_distance` and `verify_code` for exhaustive code verification.
  - `build_mds_7_4_2_3` — locally verified `[4,2,3]_7` MDS construction
    (49 words, distance 3).
  - `build_shortened_ternary_hamming_7_4_3` — locally verified shortened ternary
    `[7,4,3]_3` Hamming construction (81 words, distance 3).
- `precision.py` — residual-precision reference functions and scale-mode guards:
  - `minimum_margin_trit_planes` using the strict integer predicate
    `2*E_max < gamma*(3^R - 1)`.
  - `ternary_ecoc_width` for plain labels and single-trit-error detection.
  - `ResidualScaleMode` enum (`GEOMETRIC_BALANCED`, `LEARNED_INDEPENDENT`,
    `OTHER`) plus `assert_geometric_only` guard.
- `suggest.py` — `suggest_robust_arms` emits the robust coding arms feasible for a
  state-count target while excluding the disproven `(K=6,d=4)` and `(K=3,d=6)`
  arms and keeping `(K=7,d=4)` and the ternary `n=7` construction;
  `smallest_feasible_alphabet` is the Singleton-guarded companion of
  `smallest_injective_arity`.
- Additive re-exports in
  [`src/slm_training/dsl/analysis/arity/__init__.py`](../../src/slm_training/dsl/analysis/arity/__init__.py)
  (canonical CAP0-02 exports are left untouched).

## Corrected toy robust results

- `q=6, n=4, d=3` is infeasible by Singleton (`A_6(4,3) <= 36 < 41`).
- `q=7, n=4, d=3` is feasible with the locally verified MDS construction
  (49 words, min distance 3).
- Ternary `d=6` for M=41 is absent/infeasible; the feasible ternary robust arm is
  the shortened Hamming construction at `n=7, d=3`.

## Honesty boundary

- The two constructions are verified **exhaustively for their own committed
  parameters only** (`verify_code` recomputes size and minimum distance from
  scratch). No arbitrary `A_q(n, d)` solver is claimed.
- `A_3(6,3) = 38` is an externally sourced exact bound; it is not re-proven here.
- The external CAP0-01 estimates remain source-reported per
  [`calculated-arity-adaptive-precision.md`](calculated-arity-adaptive-precision.md);
  CAP0-03 adds local mathematical infrastructure, not a reproduction of those
  numbers.
- This is mathematical infrastructure only — no neural learned-code
  implementation and no model-performance claim.

## Not included (deferred follow-up)

Wiring coding-theory metadata into the emitted report (a `CodingMetadata` payload
on the CAP0-02 report) and a `scripts/analyze_grammar_arity.py` flag to attach it
are intentionally **out of scope here**. On current `main` the report is the
canonical `ExactArityReport` (post-minimisation certificate), which differs from
the retired arity stub's report contract those hooks were originally written
against. The `coding`/`precision`/`suggest` layer is self-contained and needs no
such wiring to be useful or tested; integrating a metadata payload into
`ExactArityReport` is a clean, separable follow-up.

## Verified (this change, Torch-free environment)

- `ruff check` on all changed files — passes.
- `python -m compileall` on the new modules and tests — passes.
- `pytest tests/test_dsl/test_arity_coding.py` — 11 passed.
- `pytest tests/test_dsl/test_arity_suggest.py` — 5 passed.
- `pytest tests/test_dsl/test_arity_analysis.py` (canonical CAP0-02) — 21 passed,
  unchanged from the pristine-`main` baseline (proves canonical arity is
  undisturbed).
- Package import is Torch-free (no `torch` module leaks into `sys.modules`).
- `python -m scripts.repo_policy` — ok. `git diff --check` — clean.
