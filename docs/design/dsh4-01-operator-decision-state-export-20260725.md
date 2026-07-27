# DSH4-01: operator decision-state export (SLM-386)

**Status:** fixture / wiring only.
**Claim class:** `wiring`.
**Honest verdict:** `fixture_wiring`.
**Blocked by:** SLM-385 (DSH3-17, Done) — no longer a blocker.

This change reuses the repository's existing legal-set trace/export contracts
(DSH3-06 `OperatorLegalSetV1`, SPV2-03 `TeacherTraceManifest`) rather than
building a duplicate distillation data stack. It adds one new module,
`src/slm_training/harnesses/distill/operator_decision_state.py`, that binds
live `CompilerState` (`OperatorStateV1`) and `ActionEffectV1` operator
candidates to accepted-set, provenance, acquisition (gold vs on-policy), and
replay contracts that already exist in this repo's legal-action fixtures.

## What changed

* `src/slm_training/harnesses/distill/operator_decision_state.py`
  * `OperatorDecisionStateTraceV1` — one certified decision-state export.
    Composes, does not duplicate:
    * `OperatorLegalSetV1` (DSH3-06) for the exact state fingerprint, complete
      legal action/operator set, coverage, and bounded verifier
      (rejection-count/rejection-sample) outcomes.
    * `ReferenceTableV1` (DSH3-03) for the inference-visible compiler-fact
      context bound to the same state.
    * `OperatorStateV1` / `ApplicationProvenanceV1` (DSH3-01/DSH3-02) for the
      compiler state and provenance captured *before* any repair or fallback
      mutation.
    * `TeacherTraceManifest.manifest_id` (SPV2-03) for the scoring-run
      provenance envelope binding — only the id is carried, exactly as
      `LegalSetTeacherTrace` already does.
    * `SupervisionSource` (EFS3-01) for gold-state vs on-policy topology/
      decoder state provenance.
  * New fields only DSH4-01 adds: `accepted_application_ids`,
    `current_scores`, `approximate`, and `capture_stage`
    (`CaptureStage.PRE_REPAIR`, the only accepted value today — the enum
    stays open rather than a bare bool so a future capture stage cannot
    silently satisfy today's acceptance bar without an explicit new member).
  * `capture_operator_decision_state()` — calls DSH3-06
    `enumerate_operator_legal_set` directly on the caller-supplied state (no
    repair/fallback path runs anywhere in this module), so every trace is
    structurally captured pre-repair/pre-fallback.
  * `OperatorTrajectoryV1` / `OperatorTrajectoryStepV1` — an ordered, chained
    sequence of accepted (proof-bearing) `OperatorApplicationV1` records.
  * `replay_operator_trajectory()` — replays a trajectory through the sole
    DSH3-02 replay authority (`OperatorLibraryV1.replay`), stops at the first
    divergent action, and classifies `OperatorReplayError`'s static messages
    into stable `operator.replay.*` codes without re-deriving replay identity
    itself.
  * `export_for_teacher_query()` — the stop-rule gate: when an accepted
    trajectory is supplied and does not replay to a verifier-equivalent
    canonical result, this raises `OperatorDecisionStateError` and returns
    nothing queryable. No hidden reasoning or chain-of-thought is ever stored
    on the export — every field is a typed compiler identity/digest, a
    bounded verifier outcome, or a numeric score.
  * `write_operator_decision_state_traces()` — JSONL export. Write-only:
    `OperatorLegalSetV1` itself has no `from_dict` (legal actions require
    re-resolving opaque references against a live `ReferenceTableV1`, a
    pack-bound operation), so this module keeps that same boundary rather
    than inventing a parallel deserialization path.
* `scripts/run_dsh4_01_operator_decision_state_fixture.py` — deterministic
  fixture runner over a tiny two-value `openui` pack operator
  (`hero.title -> hero.body -> hero.subtitle`). Captures one `GOLD` trace at
  state0 and one `ON_POLICY` trace at state1, builds the accepted two-step
  trajectory, replays it against the original registry (verified) and a
  deliberately drifted registry (diverges at step 1), and demonstrates
  `export_for_teacher_query` succeeding in the first case and being blocked
  by the stop rule in the second.
* `tests/test_harnesses/distill/test_operator_decision_state.py` — 12 tests
  covering: exact state/legal-set binding, accepted/scored id containment,
  gold + on-policy support, explicit partial-coverage/approximate evidence,
  successful two-step trajectory replay to a canonical result, first-
  divergent-action reporting on a genuinely diverged trajectory, the stop-
  rule gate (blocked and allowed paths), two fail-closed construction
  negatives (stale reference table, legal set not bound to its reference
  table), JSONL export, and rejection of a trajectory step built from a
  rejected (non-accepted) application.
* `src/slm_training/resources/versions.json` — bumped `harness.distill` to
  `v3` (adds the new module/script/tests/docs to its watched paths).

## Acceptance checklist (from the Linear issue)

* **Traces capture state/action evidence before repair or fallback
  mutation.** `capture_operator_decision_state()` calls
  `enumerate_operator_legal_set` directly on the caller's state; this module
  imports no repair (`semantic_repair.py`) or decode-fallback path, so every
  trace is structurally pre-repair/pre-fallback. `CaptureStage.PRE_REPAIR` is
  asserted in `__post_init__`.
* **Every accepted trajectory replays to a verifier-equivalent canonical
  result.** `test_accepted_trajectory_replays_to_canonical_result` builds a
  real two-step trajectory via `OperatorLibraryV1.apply` and confirms
  `replay_operator_trajectory` reproduces the identical canonical final
  state. `test_diverged_trajectory_reports_first_divergent_action` confirms a
  genuinely diverged trajectory (registry changed between recording and
  replay) reports its first divergent action instead of silently succeeding.
* **Existing trace schemas are reused or explicitly versioned, not
  duplicated.** `OperatorLegalSetV1`, `ReferenceTableV1`, `OperatorStateV1`,
  `ApplicationProvenanceV1`, `TeacherTraceManifest`, and `SupervisionSource`
  are embedded/bound by reference, not re-implemented. The new schema
  (`dsh4-01.v1`) is a distinct, explicitly versioned envelope around them.
* **Exact versus approximate evidence and coverage are explicit.**
  `OperatorDecisionStateTraceV1.coverage` (delegates to
  `legal_set.coverage: LegalSetCoverage`) and `.approximate: bool` are both
  first-class fields; `test_partial_coverage_and_approximate_flags_stay_explicit`
  exercises budget-truncated (`PARTIAL`) coverage alongside `approximate=True`.

## Stop rule

> If exact state/action identity cannot be replayed, repair the exporter
> before querying any teacher.

`export_for_teacher_query()` implements this literally: it calls
`replay_operator_trajectory()` first and raises `OperatorDecisionStateError`
— returning nothing a teacher could be queried with — whenever replay does
not verify. `test_export_for_teacher_query_blocks_on_unverified_trajectory`
and the fixture script's `stop_rule_message` both exercise this path.

## Fixture run

```bash
python -m scripts.run_dsh4_01_operator_decision_state_fixture \
  --run-id dsh4-01-20260725-fixture
```

Evidence: [dsh4-01-operator-decision-state-export-20260725.json](dsh4-01-operator-decision-state-export-20260725.json).
Output: `outputs/runs/dsh4-01-operator-decision-state/dsh4-01-20260725-fixture/`
(`summary.json`, `decision_state_traces.jsonl`).

| Metric | Value |
| --- | --- |
| Gold trace legal actions | 1 (coverage `complete`) |
| On-policy trace legal actions | 1 (coverage `complete`) |
| Trajectory steps | 2 |
| Replay verified (original registry) | `true`, 2/2 steps |
| Replay verified (drifted registry) | `false`, diverges at step 1 |
| First divergence code | `operator.replay.application_identity_mismatch` |
| Export succeeds (verified trajectory) | `true` |
| Export blocked by stop rule (drifted registry) | `true` |

## Honest caveats

* **Wiring-only fixture.** No external teacher model is downloaded or
  scored, no checkpoint is loaded or trained, and no ship gate is evaluated
  or weakened. `current_scores` in the fixture are hand-set floats, not real
  teacher output.
* **Small deterministic compiler operator.** The fixture operator has one
  argument slot and two candidate values; it demonstrates the export/replay
  contract, not model quality or coverage at production DSL scale.
* **No round-trip loader.** `write_operator_decision_state_traces` is
  write-only by design (see above); a full JSONL loader would need to
  re-resolve opaque references against a live `ReferenceTableV1`/pack, which
  `OperatorLegalSetV1` itself does not support today either.
* **Real teacher-query integration is a follow-up.** `export_for_teacher_query`
  is the fail-closed gate a real SPV2-03/SLM-108 teacher-scoring caller would
  sit behind; no such caller is wired in this change.

## Verification commands

```bash
python -m pytest tests/test_harnesses/distill/test_operator_decision_state.py -q
python -m pytest tests/test_dsl/test_operator_legal_set.py tests/test_harnesses/distill/ \
  tests/test_evals/test_dense_teacher_mixture.py tests/test_evals/test_solver_state_supervision.py -q
python -m scripts.run_dsh4_01_operator_decision_state_fixture --run-id dsh4-01-20260725-fixture
ruff check src/slm_training/harnesses/distill/operator_decision_state.py \
  scripts/run_dsh4_01_operator_decision_state_fixture.py \
  tests/test_harnesses/distill/test_operator_decision_state.py
python -m scripts.verify_version_stamps --check
```

All commands passed on this branch at the time of writing (see the final
report for exact pass/fail counts).
