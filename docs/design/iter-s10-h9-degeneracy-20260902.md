# S10 / N9: H9's "distance-to-valid-program" magnitude target is degenerate under I6

**Status: `closed_ill_posed`** — H9 is refuted numerically on the real
`CandidateRanker` seam. No model was trained, evaluated, or benchmarked; the
evidence is a deterministic CPU census that runs as an ordinary test:

- Test: `tests/test_models/test_solver_energy_h9_degeneracy.py`
  (`PYTHONPATH=$PWD/src python -m pytest tests/test_models/test_solver_energy_h9_degeneracy.py`,
  about 20 s, no randomness, no GPU, no checkpoint).
- Claim class: `fixture` (fixture-corpus census; never a ship, checkpoint, or
  production-default claim).
- Source card: `docs/design/external-analysis-audit-eqm-constrained-decode-20260902.md`
  § S10; audit row 17 (H9) and negative-result row N9.

## Hypothesis under test (external hypothesis H9)

H9 proposes extending `CandidateEnergyScorer`
(`src/slm_training/models/solver_energy.py`) with a second head trained on an
EqM-style magnitude target, "distance to the nearest valid program according to
the grammar acceptor", claiming it would reduce expanded nodes in value-guided
beam search (`src/slm_training/models/tree_edit_diffusion.py`, beam width 4,
"every emitted candidate is valid by construction").

N9 answers that the target is ill-posed on the only distribution the head would
ever see. The scorer's sole runtime authority is to *order* the exact live
candidates the solver's `CandidateRanker` seam hands it
(`src/slm_training/dsl/solver/controller.py::search`, `ranker.rank(state,
hole_id, live)` followed by `_validate_permutation`). Under invariant I6 those
live candidates are the pack's exact finite completion domain
(`OpenUIForestExpander` → `build_completion_forest`), every value of which is
admissible by construction. A grammar-acceptor distance is therefore
identically `0` across the candidates at every branch point: zero variance,
nothing to learn, and the head could only reproduce its bias.

## What the census does

For every gold `openui` program in the fixed program set, encoded exactly as
`DSLNativeTokenizer` encodes it (no renaming, no reordering):

1. Build the real `OpenUIForestExpander` for the record's slot contract and
   walk the gold program position by position from `root_state()` through
   `successor()`. At every position a recording `CandidateRanker` receives
   `(state, hole_id, live)` exactly as `search()` would call it, and the
   controller's own `_validate_permutation` check is applied to its output.
2. A *branch point* is a position whose live domain has at least two values.
3. For every live value the proposed H9 target is computed with the pack's
   completion-domain acceptor stack:
   `admit_fill` (left-prefix `InteractiveParser` admissibility),
   `multi_region_support` (exact bounded multi-region completability, with a
   budget-exhausted `unknown` counted as *not* a rejection), and the
   well-formed verifier for end-of-program candidates. The target is `0` when
   `prefix + candidate` is admissible; otherwise `1 + k` with `k` the fewest
   trailing-token deletions that restore admissibility (an upper bound on the
   edit distance; it exists only so that a violation would be *measurable*).
4. The population variance of the target across the live candidates is
   required to be exactly `0.0` at every branch point. A branch point with a
   non-constant target is documented (prefix, candidates, per-candidate
   admissibility) as an I6 legality-bug candidate via a `LegalityBugCandidate`
   warning instead of being hidden behind a bare assertion.
5. The one non-degenerate reformulation — distance to the *gold* program — is
   computed alongside for contrast (`0` iff the candidate continues the gold
   token stream, else `1`).
6. The target is shown to be non-vacuous on a planted illegal candidate
   (`) )` after `<BIND_0> = Stack`), which scores `> 0`.

Two authorities are walked: the horizon-free pack path (`remaining_tokens =
None`, the offline analysis path of `build_completion_forest`) supplies most
branch points; the strict live-decoder path (`remaining_tokens` set, so every
candidate carries a horizon-bearing terminal-witness certification) is walked
for two programs as a cross-check because it costs about a second per branch
point.

Program set (file order, fixed): all 24 records of
`src/slm_training/resources/data/eval/e938_role_safe_all_targets_smoke24_v1/suites/smoke/records.jsonl`,
the first 8 records plus `train_separator_01` of
`src/slm_training/resources/data/train/wf_smoke_v2/records.jsonl`, and the
strict-horizon walks of `smoke_empty_01` and `smoke_callout_01`.

## Measured results

| quantity | value |
| --- | --- |
| programs walked | 35 (33 pack path, 2 strict horizon) |
| ranker seam calls (= positions, singletons included) | 682 |
| **branch points (≥ 2 live candidates)** | **475** (454 pack path, 21 strict horizon) |
| branch points where the gold continuation is live | 461 |
| live candidates scored | 7 071 |
| mean candidates per branch point | 14.89 (min 2, max 64) |
| candidate-count distribution | 233 × 2, 11 × 3, 13 × 4, 19 × 5, 8 × 11, 14 × 12, 82 × 32, 71 × 33, 5 × 63, 11 × 64, 9 others |
| **H9 target variance, max over branch points** | **0.0** |
| branch points with any non-zero H9 target | 0 |
| candidates failing `admit_fill` | 0 |
| candidates failing `multi_region_support` (or `unknown`) | 0 |
| legality-bug candidates | none |
| distance-to-gold variance, min over gold-live branch points | 0.0154 (> 0 at all 461) |
| exactly one gold candidate per gold-live branch point | yes (461 / 461) |

Every one of the 7 071 candidates that reached the seam was admissible under
both acceptors and the well-formed verifier where applicable; the H9 target was
`0` for all of them and its variance was `0.0` at all 475 branch points. The
strict-horizon walks (horizon-pruned domains: 20 instead of 32 root components
for `smoke_empty_01`) behave identically.

## Closure

**H9 is closed as ill-posed under I6.** The proposed magnitude target is a
constant on the candidate stream the `CandidateRanker` seam produces, so there
is no gradient, no ordering information, and no expanded-node reduction to be
had: a head trained on it learns its bias. This is a property of the seam, not
of the corpus size — I6 guarantees it for every legal branch point, and the
census confirms the guarantee holds on the implementation. A rejected
experiment closes an approach, never a goal.

The one non-degenerate reformulation is **distance to the gold program** —
positive variance at every gold-live branch point (min 0.0154, exactly one
zero-distance candidate each). That is the existing VSS3-02 search cost-to-go
target `CandidateEnergyScorer` already learns from replay-verified
`candidate_cost` rows (`WORK_TARGET_VERSION = "v1"`,
`cost_target_from_row`), not a second head. Nothing in this record motivates
a parameter increase (`EG_params` is not engaged: no growth is proposed).

## Observations recorded, out of S10 scope

These are *coverage* (completeness) facts about the expander's canonical form
versus the fixture serialization. None is a legality (I6 soundness) matter:
the expander enumerates a strict subset of valid programs, and every walk that
left the gold path did so because the gold continuation was absent from the
domain, never because an illegal candidate was present. The test names each
divergence and fails on any *unclassified* one, so a new divergence cannot
hide behind these.

1. `bind_statement_order` (13 of 33 pack-path walks). The expander numbers
   binds by first mention and declares pending binds in mention order; the
   fixture (and `dsl/canonicalize.py`, which is topological first-use with
   children defined before parents) numbers by definition order. Example:
   `smoke_button_01` gold `<BIND_0> = Stack ( [ <BIND_2> ] )` where the domain
   offers only `bind_reference_root_children:<BIND_1>`. The 21 walks that
   reached the terminal are exactly those whose fixture order coincides with
   mention order.
2. `defaults_elided_argument_list` (1 walk, `train_separator_01`). The
   sanitize pass records `defaults_elided` and the fixture keeps
   `Separator()`; after `Separator (` the domain offers only
   `lit:STR:horizontal` / `lit:STR:vertical`, never `)`.
3. Strict-horizon witness conservativeness. With `remaining_tokens = gold
   length + slack`, the horizon-bearing certification declares the gold path
   DEAD (`illegal_prefix`, a "none"-coverage forest) for `smoke_empty_01` at
   slack ≤ 4 (terminal at 6), for `smoke_login_01` at slack ≤ 2 (terminal at
   4), while `smoke_callout_01` walks at slack 1. DEAD is a refusal, never an
   illegal candidate, so I6 is untouched; the test uses slack 8 and records
   the thresholds here rather than asserting them.

Each of these belongs to a separate card (fixture/canonical-form alignment,
horizon witness tightness); they are not H9 evidence in either direction.

## Reproduction

```
PYTHONPATH=$PWD/src python -m pytest tests/test_models/test_solver_energy_h9_degeneracy.py -q
PYTHONPATH=$PWD/src python -c "import sys, json; sys.path.insert(0, 'tests/test_models'); import test_solver_energy_h9_degeneracy as m; s = m.census_summary(m.run_census()); s.pop('walks'); print(json.dumps(s, indent=1))"
```

No JSON mirror is written for this record: it carries no run artifact, no
metric under a version-stamped harness, and no checkpoint; the numbers above
are regenerated by the test on every CI run.
