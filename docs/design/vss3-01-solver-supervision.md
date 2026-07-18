# VSS3-01: Replay-verified on-policy solver supervision

**Issue:** SLM-69  
**Status:** wiring / fixture evidence. No train, eval, benchmark, model, checkpoint, or ship claim.

## What was added

A solver supervision corpus builder that turns solver traces into versioned
`support_set` and `candidate_cost` rows.

- `src/slm_training/harnesses/distill/solver_supervision.py`:
  - `SolverTrace` envelope with `SearchResult`, `certificate_store`, support events,
    state snapshots, and lineage.
  - `SupportSetRow` and `CandidateCostRow` dataclasses with `to_dict` round-trips.
  - `ProviderRegistry` for `(pack_id, constraint_version)` → expander/verifier bundles.
  - `build_solver_supervision(config)`:
    - Reads `kind: solver` traces from a `TraceStore`.
    - Replays every certificate with `replay_support_certificate` when
      `verify_replay=True` and a registry is supplied.
    - Emits one candidate-cost row per support event and one aggregated support-set
      row per `(state_fingerprint, hole_id)`.
    - Skips tampered/unreplayable traces with structured rejection reasons.
    - Supports dry-run and immutable modes.
- `scripts/build_solver_supervision.py`: thin CLI mirroring `build_train_data.py`
  conventions (`--trace-root`, `--output-root`, `--version`, `--verify-replay`,
  `--dry-run`, `--immutable`).
- `src/slm_training/data/store.py`: added `solver_supervision` as a local-only
  `DataKind`.
- `tests/test_harnesses/distill/test_solver_supervision.py`: 6 regression tests with
  a fake finite support oracle, covering support-set/candidate-cost emission,
  unknown values, replay rejection of tampered certificates, preservation of all
  supported alternatives, dry-run, and `DataStore.resolve`.

## Verified

- `ruff check` passes.
- `python -m compileall` passes.
- `pytest tests/test_harnesses/distill/test_solver_supervision.py tests/test_data_store.py -q` → 9 passed.
- `python -m scripts.repo_policy` ok.
- `git diff --check` clean.
- `.githooks/check-changed` passes for the changed files; one unrelated pre-existing
  failure in `tests/test_data/test_verify.py::test_preview_runtime_and_behavior_seeded_failures`
  (missing `src/src/.../preview.js` build artifact) is outside this change.

## Design boundaries preserved

- Every hard support label is tied to a `SupportCertificate`.
- Unknown values are not silently converted to negative labels.
- Cost rows are emitted only from observed support events; there is no fabricated
  cost for missing suffixes.
- Lineage (`program_family_id`, `lineage_id`, `split_group_id`, `split`) is copied
  from the source trace and never mixed across splits.
- `solver_supervision` is excluded from automatic Git publication.

## Caveats

- This is builder wiring only. Real solver traces still need a live solver trace
  recorder (VSS1-04 extension) and pack-provided expander/verifier bundles.
- The CLI defaults to certificate-based emission because it has no generic solver
  providers; programmatic callers must supply a `ProviderRegistry` for replay.
- No model, checkpoint, eval run, or ship gate is claimed.
