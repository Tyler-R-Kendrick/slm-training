# KERN-04 — Black-box UNSUPPORTED query lower bound (SLM-522)

## Claim

Under the named machine/query model `BlackBoxVerifierQueryModel` — a finite
complete domain of `P` assignments where each query returns one Boolean
validity bit for one complete assignment — any deterministic **sound**
refuter that concludes `UNSUPPORTED` must query all `P` assignments in the
worst case.

If it stops after querying a proper subset, an unqueried index `i` yields a
one-valid world that agrees with the all-invalid world on every queried
answer, so the transcript cannot soundly certify unsupportedness.

This is a **model-relative** lower bound, not a universal solver lower bound.

## Preconditions (explicit)

1. **Finite complete domain** of `P` assignments (sizes may come from
   KERN-02/03 proved domains: `P = ∏ d_i`).
2. **Black-box Boolean answers only** — one validity bit per complete
   assignment; no shared intermediate state in the query alphabet.
3. **Deterministic sound refuter** — if the strategy outputs UNSUPPORTED, every
   oracle consistent with the transcript must be all-invalid.
4. **Cost model** — `BlackBoxVerifierQueryModel` query counts only (not
   wall-clock).

## Lean

- Module: `LeverProofLean.BlackBoxUnsupportedLowerBound`
- Model: `BlackBoxVerifierQueryModel` / `worstCaseQueryLowerBound = P`
- Gap lemma: `unqueried_oracles_agree`, `early_stop_distinguishes`
- Coverage: `sound_unsupported_requires_full_coverage`
- Main theorem: `black_box_unsupported_query_lower_bound`
- Escapes named: `EscapeClass` + `escapes_leave_black_box_model`

## Python

- Adapter: `slm_training.formal.black_box_unsupported_lower_bound`
- Exhaustive finite-model checks for small `P`
  (`resources/formal/black_box_unsupported_fixtures.v1.json`)
- Four-axis export: `export_four_axis_lower_bound_evidence` →
  `RevmathFourAxisAnalysisV1` with `resource_bounds` on
  `bound.black_box.unsupported_query_lower.v1`

## What can and cannot evade the bound

| Class | Evades? | Why |
| --- | --- | --- |
| Black-box per-assignment Boolean probes | **No** | Inside the model; theorem applies |
| Symbolic constraints / region queries | **Yes** | Different query alphabet; leaves the model |
| Certificates / proof objects | **Yes** | Extra structure beyond one Bool per leaf |
| Shared residual / DFA memoization | **Yes** | Correlates answers across assignments |
| Grammar / prefix structure pruning | **Yes** | Rejects many leaves without leaf queries |
| Wall-clock / device scheduling | N/A | Never claimed (empirical remainder) |

Escapes evade the **assumption**, not the theorem. Inside
`BlackBoxVerifierQueryModel` the lower bound stands.

## Empirical boundary

Query lower bounds certify combinatorial cost under the named model. Runtime
VSS schedulers, residual caches, and certificate-producing solvers are
implementation-refinement remainders — they are outside this black-box claim
by construction.
