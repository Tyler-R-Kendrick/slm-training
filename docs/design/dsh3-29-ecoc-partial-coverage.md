# DSH3-29 cost-aware ternary ECOC abstention under controlled PARTIAL coverage

SLM-404 tests whether the DSH3-03/CAP2-03 `TernaryECOCHead`'s cost-aware
codeword assignment (`build_ternary_ecoc_entry(..., costs=...)`) gives a real
selective-risk benefit over uniform assignment once the legal action set is
truncated to a controlled PARTIAL coverage fraction, and whether ECOC's model
abstention is honestly separated from compiler-level PARTIAL/UNKNOWN defer.
It does not change legal-set enumeration, hard pruning, singleton forcing, or
decoder routing.

## Cost matrix

`build_operator_action_cost_matrix` (`src/slm_training/models/semantic_cost.py`)
builds a symmetric, deterministic pairwise cost from three
`OperatorActionViewV1`-only signals: absolute declaration-cost delta,
`EffectDeltaKind` symmetric-difference size (the compiler-visible stand-in for
canonical-AST-cost divergence -- the sanitized view carries no raw AST), and a
locality-mismatch indicator. It reads no field outside
`OperatorActionViewV1`, so it structurally cannot reference an after-state,
proof, or target field banned by `operator_policy_view.FORBIDDEN_FIELD_NAMES`.

## Coverage ladder

`src/slm_training/models/operator_ecoc_coverage.py` adds:

* `partial_coverage_levels(total)` -- the 100/75/50/25% ladder as integer
  budgets, monotonic non-increasing and de-duplicated for small `total`.
* `build_partial_ladder(objective, cost_matrix, seed=0)` -- reuses DSH3-25's
  `build_controlled_partial_fixture` shadow-set machinery (never re-derives
  truncation) to build one `ControlledPartialFixtureV1` per level. Each
  level's truncation order keeps the *lowest* average-pairwise-cost actions
  longest, so the highest-cost-of-confusion actions are hidden first -- at low
  coverage only cheap, easy-to-distinguish actions remain witnessed.
* `evaluate_ecoc_under_coverage(head, ladder, cost_matrix)` -- classifies each
  level's decode outcome (against the shadow's gold action versus the
  highest-cost-of-confusion decoy still present) into exactly one of:
  `compiler_defer_count` (gold was truncated out of the public view --
  compiler-level PARTIAL/no-witness, never a model decision),
  `model_abstain_count` (`abstain`/`refine`), `detected_errors`,
  `retained_wrong_actions` (a silently wrong scored decision), an
  `invalid_codewords` fallback-guess bucket, or `correct`. `defer_count` is
  their sum, reported alongside the two components so model uncertainty is
  never conflated with compiler UNKNOWN/PARTIAL. Cost-weighted and plain risk
  are both reported per level and overall, plus a selective-risk AURC
  (wrapping the existing DSH4-02 `selective_risk_aurc`).

The decode-ambiguity probe corrupts the gold action's codeword by 2 trits
toward the decoy's codeword -- the smallest probe that can expose residual,
assignment-dependent confusability once the distance-2 code's guaranteed
single-trit detection is exhausted.

## 1st result (2026-07-26)

Fixture: one small synthetic four-action set (`alpha_catastrophic`,
`beta_catastrophic` -- disjoint two-element `EffectDeltaKind` signatures, cost
8.0 to confuse; `delta_safe`, `gamma_safe` -- cost 0-4 to confuse with anyone),
with the gold action rotated across all four to give four synthetic states
from one fixture. Report:
[`dsh3-29-ecoc-partial-coverage-2026-07-26/report.json`](dsh3-29-ecoc-partial-coverage-2026-07-26/report.json),
[`summary.md`](dsh3-29-ecoc-partial-coverage-2026-07-26/summary.md).
Reproduce: `python scripts/run_dsh3_29_ecoc_partial_coverage.py --output-dir docs/design/dsh3-29-ecoc-partial-coverage-2026-07-26`
(runs in ~2.4s, far under `MAX_RUN_MINUTES`).

At full (100%, still PARTIAL-labeled) coverage the cost-aware assignment
separates the catastrophic pair to Hamming distance 3, so the 2-trit probe is
caught as `detected_error`; a uniform assignment happens to leave that pair at
the code's minimum distance of 2, so the identical probe silently reproduces
the decoy's own codeword (`retained_wrong_action`) -- see
`tests/test_models/test_operator_ecoc.py::test_cost_aware_assignment_reduces_cost_weighted_risk_under_partial_coverage`
for the isolated, deterministic single-level version of this mechanism.
Aggregated over all four rotated states and all four coverage levels,
cost-aware overall cost-weighted risk (1.00) is strictly below uniform's
(1.75); plain (unweighted) risk is roughly comparable between the two arms,
showing the benefit is specifically in *which* errors get caught, not in the
overall error count. `LocalFlatHead` is reported only as a coverage-structure
baseline (gold-witnessed / forced-singleton counts per level) since it has no
codeword geometry to compute a comparable cost-weighted risk against.

**Decision: `fixture-scale-positive-signal`.** This is real, deterministic
evidence that cost-aware ECOC codeword assignment can convert a silently
wrong catastrophic decode into a caught `detected_error` under controlled
PARTIAL coverage -- but it is fixture-scale wiring evidence on one hand-built
four-action set, not a ship claim. Per DSH3-28's own disposition, natural
PARTIAL-coverage rows remain too rare in the real `operator_policy_corpus` to
run this at scale today. This result does **not** overturn the standing
DSH3-28 stop rule ("if ECOC abstentions are uncorrelated with error cost or
add no selective-risk benefit, retain the simpler best DSH3-28 head") --  it
is one positive fixture-scale data point in favor of *keeping* cost-aware
ECOC as a live candidate, not evidence sufficient to promote it. The
re-test trigger is unchanged from DSH3-28: re-run once
`operator_policy_corpus` supplies natural PARTIAL rows at meaningful volume.

## Scope

This is a unit-tested schema/coverage/evaluation contract plus one small
synthetic-fixture script run, not a train, eval, matrix, checkpoint, or ship
claim.
