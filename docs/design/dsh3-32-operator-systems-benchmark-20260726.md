# DSH3-32 (SLM-407) — operator-inference systems benchmark (2026-07-26)

Linear: SLM-407 · Evidence: [`dsh3-32-operator-systems-benchmark-20260726.json`](dsh3-32-operator-systems-benchmark-20260726.json)

## Decision this issue enables

Whether the trained operator path (typed policy + singleton bypass) reduces
real forward-equivalent work relative to full generation, X22, serialized
actions, and a compiler-only forced baseline, after charging legal-set
construction and dry-run costs to every arm.

## Scope decision (read before the results)

This is a **fixture-scale wiring harness**, in the same family as
`evals/tree_edit_scaling.py` and `evals/cap2_operator.py`. It loads no
trained checkpoint and makes no quality or ship claim.

**Five of six named arms are measured for real:**

* `compiler_only_forced` — deterministic compiler enumeration + first-in-order
  serialized action, zero model forwards.
* `serialized_actions` — enumerate and emit the full candidate wire format;
  nothing committed, zero model forwards.
* `typed_policy_singleton_bypass_on` — `decide_typed_operator_policy`
  unmodified: a true COMPLETE_SINGLETON state costs zero forwards.
* `typed_policy_singleton_bypass_off` — the same scorer, but forced to score
  every COMPLETE state (including true singletons), to measure the forward
  the singleton-bypass invariant is saving in practice. PARTIAL coverage
  still never forces, regardless of this toggle (a separate, non-negotiable
  invariant this benchmark does not touch).
* `x22_tree_edit` — one real `TreeEditDiffusionModel` decode; forward and row
  counts come from new counters added to the model's own evidence dict
  (`_decode_one`'s `self.policy(...)` call sites), not an estimate.

**`full_generation` is explicitly left unrun.** The issue's own acceptance
text names a sixth arm: plain end-to-end causal-lm decode with no operator
legal set at all. Wiring that honestly needs a trained/checkpointed causal-lm
decode path that does not exist at fixture scale here; fabricating a number
for it would violate this repo's honesty law. It is recorded in
`OperatorServingBenchmarkV1.arms_unrun`, not silently dropped.

## What was added

* `src/slm_training/evals/operator_systems_benchmark.py` — `OperatorServingWorkV1`
  (the systems numeraire: `model_forwards`, `policy_rows_scored`, `dry_runs`,
  `executor_calls`, `validator_calls`, `wall_ms`, `cpu_ms`, plus
  `legal_set_size`/`operator_row_count`/`coverage`/`cache_state` for
  stratification) and `OperatorServingBenchmarkV1` (the disposition report:
  `arms_measured`, `arms_unrun`, `claim_class="wiring"`, `ship_eligible=False`).
* `src/slm_training/models/tree_edit_diffusion.py` — two new counters
  (`forwards`, `forward_rows`) in `_decode_one`'s evidence dict, incremented
  at both `self.policy(...)` call sites. Additive only; no other behavior
  changed. `TreeEditDiffusionModel` and `dsl.operators.legal_set` are the two
  other symbols this issue's own dependency map (`evals/cap2_operator_policy_rebase.py`)
  names as DSH3-32's target surface, alongside `models.decode_stats`.
* `scripts/run_operator_systems_benchmark.py` — CLI that runs the harness and
  writes `report.json` + `summary.md`.
* `tests/test_evals/test_operator_systems_benchmark.py` — regression tests
  (see Hard gates).

## Fixture strata

Three real operator legal-set fixtures via `enumerate_operator_legal_set`
over the `openui` DSL pack (`build_operator_fixture`):

| Stratum | Setup | Route |
| --- | --- | --- |
| `singleton` | 1 registered operator, 1 accepted value | typed policy: `COMPLETE_SINGLETON` |
| `ambiguous` | 2 registered operators, 2 accepted values | typed policy: `COMPLETE_AMBIGUOUS` |
| `partial` | 1 operator, `repeated=True` slot | `PARTIAL` (unbounded repeated-slot, never dry-run) |

Plus one `x22_tree_edit` stratum (`tree_edit_search`) with `cache_state="n/a"`
since legal-set caching does not apply to X22's own search.

Each of the four legal-set arms runs `cold` (fresh `enumerate_operator_legal_set`)
then `warm` (reused `OperatorLegalSetV1`, `dry_runs=0`) per stratum, per
`--repeats` (default 2, evidence run used 3).

## Fixture run (2026-07-26)

`python scripts/run_operator_systems_benchmark.py --repeats 3`, aggregated
across the run's 9 legal-set rows per arm (3 strata × 3 repeats) and 3
`x22_tree_edit` rows:

| Arm | Rows | Σ model_forwards | Σ dry_runs | Σ wall_ms | Failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `compiler_only_forced` | 9 | 0 | 9 | 29.1 | 3 (the `partial` stratum has no forced action) |
| `serialized_actions` | 9 | 0 | 9 | 17.9 | 0 |
| `typed_policy_singleton_bypass_on` | 9 | 3 | 9 | 30.5 | 0 |
| `typed_policy_singleton_bypass_off` | 9 | 6 | 9 | 32.9 | 0 |
| `x22_tree_edit` | 3 | 24 | 0 | 205.5 | 0 |

The core finding: **bypass-on costs 3 forwards, bypass-off costs 6, over the
same 9 rows.** The extra 3 forwards are exactly the 3 repeats of the
`singleton` stratum being scored when the toggle forces them — the
`ambiguous` stratum scores under both toggles (`COMPLETE_AMBIGUOUS` always
forwards) and `partial` never forwards under either toggle. This is measured,
not assumed: `tests/test_evals/test_operator_systems_benchmark.py` asserts it
per-row and end-to-end.

`compiler_only_forced`'s 3 failures are the 3 `partial`-stratum rows: a
repeated-slot operator produces zero dry runs (the enumerator never attempts
an unbounded product) and zero serialized actions, so there is nothing to
force. That is a correct measurement of the "repeated slot unbounded"
short-circuit, not a bug — see `test_compiler_only_forced_repeated_slot_never_dry_runs`.

Warm cache rows show `dry_runs=0` (no re-enumeration) with the same
`actions_committed` as their cold counterpart, confirming legal-set/
reference-table caching is measured, not asserted.

## Honest caveats

* **`executor_calls`/`validator_calls` are 0 for the compiler-side arms.**
  `OperatorLibraryV1.dry_run` and `.apply` both delegate to the same private
  `_execute`; this repo does not separate an "executor" call from a
  "validator" call at that layer, so this benchmark does not fabricate that
  split. `validator_calls=1` is only populated for `x22_tree_edit`, where
  `generate_batch_requests` really does call `dsl.parser.validate` once per
  decode.
* **Only legal-set/reference-table caching is measured.** The issue names
  warm/cold prompt caching, batching, prefix sharing, and materialization as
  separate axes to instrument. No arm in this repository implements those
  cache layers today for operator inference, so they are not measured here —
  an explicit gap (see `arms_unrun` reasoning above; the gap itself is not
  listed there because it is a caveat on measured arms, not an unrun arm).
* **`x22_tree_edit`'s scorer is untrained** (fresh `TreeEditDiffusionModel`,
  random init, `d_model=16`). Forward/row counts are real; the decoded output
  quality is not evaluated and carries no claim.
* **Not a crossover fit.** The issue's stop rule asks for a measured
  crossover between arms; this fixture-scale pass establishes the numeraire
  and per-arm instrumentation. A crossover claim needs real workload-scale
  strata (state size, sequence length) this harness's three small fixtures do
  not attempt to represent.
* **Single-process wall/cpu timings on this run's shared CPU host** — not a
  pinned hardware/software identity. Comparative `wall_ms` numbers here are
  illustrative of the harness working, not a systems-efficiency claim.

## Hard gates

* `pytest -q tests/test_evals/test_operator_systems_benchmark.py tests/test_models/test_decode_stats.py tests/test_dsl/test_operator_legal_set.py tests/test_harnesses/model_build/test_eval_gates.py tests/test_models/test_tree_edit_diffusion.py tests/test_harnesses/experiments/test_typed_operator_policy.py` — 105 passed.
* Singleton-bypass invariant confirmed at the row level
  (`test_singleton_stratum_bypass_on_costs_zero_forwards`,
  `test_singleton_stratum_bypass_off_forces_one_forward`) and end to end
  (`test_run_operator_systems_benchmark_end_to_end_small`).
* PARTIAL-never-forces invariant confirmed independent of the bypass toggle
  (`test_partial_stratum_never_forces_regardless_of_bypass_toggle`).
* `compiler_only_forced` never invokes the model on any stratum
  (`test_compiler_only_forced_never_invokes_the_model`,
  `test_compiler_only_forced_repeated_slot_never_dry_runs`).
* Warm-cache dry-run elimination confirmed
  (`test_warm_cache_state_skips_re_enumeration_dry_runs`).
* `python -m scripts.verify_version_stamps --check` — passes; new component
  `evals.operator_systems_benchmark` (v1) registered, and
  `harness.experiments.slm299_edit_reachability` (the pre-existing owner of
  `tree_edit_diffusion.py`) bumped to v4 for the additive evidence-counter
  change.

## Disposition

**Measured, not shipped.** `compiler_only_forced` and
`typed_policy_singleton_bypass_on` are confirmed to cost zero learned-model
forwards on singleton/repeated-slot states, and the benchmark shows exactly
how many forwards the singleton-bypass invariant removes relative to a
bypass-off control on the same rows. `full_generation` is explicitly unrun.
No crossover, ship-gate, or systems-efficiency claim is made from this
fixture-scale pass; `OperatorServingWorkV1`/`OperatorServingBenchmarkV1` give
DSH3-33 (SLM-408, the follow-on disposition issue) a typed numeraire to bind
a real crossover claim to once workload-scale strata exist.

## Stop rule

Not triggered: the harness runs and every measured arm behaves per its
documented invariant. Nothing here supports an inference-efficiency claim on
its own; `full_generation` remaining unrun means a crossover verdict against
plain generation cannot be issued from this evidence.
