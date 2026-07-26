# DSH5-06 bounded set-valued transaction policy: wiring result, not a ship claim

SLM-414 (DSH5-06) asks whether one model forward can select a small, exact,
conflict-free action **set** more accurately and efficiently than sequential
single-action policy calls, while preserving atomic compiler authority. This
document reports what was actually built and measured:
`src/slm_training/harnesses/experiments/operator_transaction_policy.py`, its
tests, and a two-example fixture-scale control matrix. It is **fixture/wiring
evidence only** — status matches the "Status: bounded local measured result;
not a ship claim" convention used by the sibling DSH3-28 report, scaled down
further (N=2, not a real corpus).

## Decision and hypothesis

Per the issue: a bounded independent-set or recurrent-set head over prepared
base-state actions, masked by the exact base-state conflict graph, should
reduce ambiguous policy rounds for multi-edit requests without weakening
atomic compiler authority anywhere. The hypothesis under test here is
narrower — it is the wiring precondition for that larger question: **can a
learned per-action score be used to rank candidate bounded (K≤4) action sets
while every executed set remains exactly conflict-free, even when the
ranking's belief about which pairs conflict is deliberately corrupted?**

## What this reuses, not reimplements

* **Conflict/atomicity ground truth** — every notion of "conflict-free" in
  this module is the real
  `dsl.operators.transactions.build_operator_transaction` (DSH5-04, SLM-412)
  and every commit is the real
  `dsl.operators.transaction_executor.commit_operator_transaction` (DSH5-05,
  SLM-413). No parallel conflict-detection or composition logic was added.
* **Per-action scoring** — `harnesses.experiments.typed_operator_policy`'s
  `independent_set` and `recurrent_set` head families (DSH3-28, SLM-403)
  compute one permutation-equivariant logit per sanitized candidate action
  from an `OperatorPolicyInputV1` view built with empty `argument_slots`
  (arguments are already bound at prepare time; the transaction policy only
  chooses *which* prepared actions to combine). `TypedOperatorPolicyScorer`
  is used unmodified; this issue adds bounded-K set decoding on top of its
  existing per-action logits, not a new encoder.

## Bounded set decode: rank, then always verify before commit

`decode_bounded_transaction_set`:

1. If coverage is not `COMPLETE`, return DEFER with zero model forwards —
   PARTIAL action coverage never implies complete-set knowledge (acceptance
   criterion, enforced by `test_partial_coverage_never_force_emits_a_complete_set`).
2. Compute the ground-truth pairwise-conflict belief by literally calling
   `build_operator_transaction` on every candidate pair (a fast, exact
   necessary-condition prune — never an approximation of the real check).
3. Enumerate every subset of size 1..K surviving that prune (bounded by
   `MAX_TRANSACTION_POLICY_CANDIDATES = 8`, mirroring
   `OPERATOR_TRANSACTION_MAX_ACTIONS`), score each by the sum of its members'
   logits from **one** scorer forward call, and rank them highest-first.
4. Walk the ranked list and **re-verify each one** with the real
   `build_operator_transaction` before accepting it. A ranked subset that
   fails this real check is counted in `conflict_attempts` and never
   executed; the walk continues to the next-ranked candidate.

This is a direct instance of this repository's own non-negotiable invariant
("speculation ranks only over forward-calculated symbol tables and always
verifies before commit"): the learned score only orders a search; the
executor is the sole authority on what may commit.

### The safety invariant, stress-tested

`test_shuffled_conflict_belief_never_commits_a_real_conflict` builds two
candidates that both write the same target (a genuine conflict) plus one
disjoint candidate, then feeds the ranking step a **deliberately inverted**
conflict belief (claims the genuine conflict is fine and invents fake
conflicts elsewhere — the `shuffled_conflict_graph` control). The test
asserts: the decoder never returns the genuinely conflicting pair, the
eventually-selected set still commits successfully through the real
executor, and reaching it required at least one blocked `conflict_attempts`
event. The corrupted belief costs search efficiency; it cannot buy an unsafe
commit.

## Matrix and controls (N=2 fixture; not a corpus)

Two hand-built `openui`-pack examples (one three-candidate disjoint-write
example with an accepted 2-action set; one three-candidate example with a
genuine two-way conflict and two valid 2-action alternatives), K=2, 5 Adam
training steps, `dim=8`. Five arms per head family, exactly the issue's
required controls:

| Head | Arm | Exact-set match | Accepted-set mass | Final-state success | Conflict attempts | Model calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `independent_set` | enabled | 1.00 | 1.00 | 1.00 | 0 | 2 |
| `independent_set` | disabled (zero) | 0.00 | 0.50 | 1.00 | 0 | 2 |
| `independent_set` | random_score | 0.00 | 0.50 | 1.00 | 0 | 2 |
| `independent_set` | shuffled_conflict_graph | 0.00 | 0.50 | 1.00 | 1 | 2 |
| `independent_set` | oracle_set | 1.00 | 1.00 | 1.00 | 0 | 0 |
| `recurrent_set` | enabled | 1.00 | 1.00 | 1.00 | 0 | 2 |
| `recurrent_set` | disabled (zero) | 0.00 | 0.50 | 1.00 | 0 | 2 |
| `recurrent_set` | random_score | 0.00 | 0.50 | 1.00 | 0 | 2 |
| `recurrent_set` | shuffled_conflict_graph | 0.00 | 0.50 | 1.00 | 1 | 2 |
| `recurrent_set` | oracle_set | 1.00 | 1.00 | 1.00 | 0 | 0 |

Observations, read narrowly:

* `final_state_success_rate` is 1.0 in every arm, including
  `shuffled_conflict_graph` — the invariant under test holds at this scale:
  a corrupted ranking belief never produced an uncommittable or unsafe
  selection, only extra `conflict_attempts`.
  `enabled`/`oracle_set` reach the exact accepted set both times;
  `disabled`/`random_score`/`shuffled_conflict_graph` reach it 0/2 times
  (they instead commit a different, still-valid, single-action or
  alternate-pair fallback) — a real, matched enabled-vs-baseline separation,
  but from **two examples**, not a powered held-out claim.

## Acceptance criteria disposition

* "Every executed set is exactly conflict-free and atomic" — **held** at
  this scale, and adversarially exercised by the shuffled-belief control
  above (`test_shuffled_conflict_belief_never_commits_a_real_conflict`).
* "PARTIAL never force-emits a complete set" — **held**
  (`test_partial_coverage_never_force_emits_a_complete_set`).
* "At least 5% of eligible multi-edit decisions change and correct changes
  exceed wrong changes" / "better final-state success or fewer measured
  policy rounds" — **not evaluated**: this requires a real multi-edit
  transaction corpus (sequential-DSH3-policy baseline, bulk-only baseline,
  full-regeneration baseline) that does not exist yet for transactions. Two
  hand-built examples cannot support this claim in either direction; it is
  left `UNRUN`, not asserted positive or negative.

## Stop-rule disposition

The issue's stop rule ("if set-valued selection adds no held-out benefit or
causes transaction failures, retain sequential/bulk policies and close
learned transaction selection") **cannot be evaluated by this change** —
there is no held-out transaction corpus or sequential/bulk baseline
implementation to compare against yet. This PR delivers the wiring
precondition (bounded-K decode + the safety invariant it depends on) and
explicitly does not claim the stop rule is cleared either way. Set-valued
selection remains default-off, invoked only from this module's own
tests.

## Scope notes (deliberately deferred)

* No real transaction-policy training corpus, no sequential-DSH3/bulk-only/
  full-regeneration baseline comparison, and no CLI runner script writing to
  `outputs/` — building the multi-edit request generator and matched
  baselines described in the issue's "Matrix and controls" section is a
  separate, larger effort than this PR's wiring scope. The five-arm matrix
  above was run directly against the harness's Python API (not a script) and
  is reproducible from
  `tests/test_harnesses/experiments/test_operator_transaction_policy.py`'s
  fixtures.
* `k=4` is supported by the decode/train/evaluate API (`k` is a plain
  parameter) but the committed test fixtures only exercise `k=2`; a fourth
  candidate is used only in `test_bounded_k_is_respected` to prove the bound
  itself, not to measure a K=4 quality effect.
