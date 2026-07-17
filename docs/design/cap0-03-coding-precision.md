# CAP0-03: Exact robust-code constructions and residual-precision reference functions

**Issue:** SLM-79  
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

Extended the CAP0-02 arity analyzer with mathematical infrastructure under `src/slm_training/dsl/analysis/arity/`:

- `coding.py` — exact coding-theory reference functions and verified constructions:
  - `smallest_injective_arity`, `hamming_ball_volume`, `hamming_sphere_packing_holds`, `gilbert_greedy_guarantees`, `singleton_upper_bound`.
  - `minimum_distance` and `verify_code` for exhaustive code verification.
  - `build_mds_7_4_2_3` — locally verified `[4,2,3]_7` MDS construction (49 words, distance 3).
  - `build_shortened_ternary_hamming_7_4_3` — locally verified shortened ternary `[7,4,3]_3` Hamming construction (81 words, distance 3).
- `precision.py` — residual-precision reference functions and scale-mode guards:
  - `minimum_margin_trit_planes` using the strict integer predicate `2*E_max < gamma*(3^R - 1)`.
  - `ternary_ecoc_width` for plain labels and single-trit-error detection.
  - `ResidualScaleMode` enum (`GEOMETRIC_BALANCED`, `LEARNED_INDEPENDENT`, `OTHER`) plus `assert_geometric_only` guard.
- Exports added to `src/slm_training/dsl/analysis/arity/__init__.py`.

## Verified

- `ruff check` passes.
- `python -m compileall` passes.
- `pytest tests/test_dsl/test_arity_coding.py` passes (11 tests).
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.

## Corrected toy robust results

- `q=6, n=4, d=3` is infeasible by Singleton (`A_6(4,3) <= 36 < 41`).
- `q=7, n=4, d=3` is feasible with the locally verified MDS construction (49 words).
- Ternary `d=6` for M=41 is absent/infeasible; the feasible ternary robust arm is `d=7` via the shortened Hamming construction.

## Caveats

- `A_3(6,3)=38` is an externally sourced exact bound; no local exact solver for arbitrary `A_q(n,d)` is claimed.
- This is mathematical infrastructure only; no neural learned-code implementation or model performance claim.
