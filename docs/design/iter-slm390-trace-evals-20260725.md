# SLM-390 (DSH4-05): quarantined runtime trace-to-eval candidates

Fixture/wiring harness. Turns recurring runtime failures (W3C run traces,
`slm_training.runtime.telemetry.trace`) into **candidate** evals without
automatically contaminating training and without letting a generated verifier
certify itself. No model is trained; no GPU is required.

Module: `src/slm_training/harnesses/experiments/slm390_trace_evals.py`
Tests: `tests/test_harnesses/experiments/test_slm390_trace_evals.py`
Machine-readable result: `iter-slm390-trace-evals-20260725.json` (version-stamped).

## Lifecycle

```text
QUARANTINED -> REVIEWED -> FROZEN -> CONFIRMATION_USED
                               +-> PROMOTED_TO_TRAINING (separate transition)
```

Every transition is explicit; there are **no automatic transitions** — a
freshly ingested trace with no review evidence stays `QUARANTINED` (tested).

- **Ingest (`ingest_trace`)** redacts/de-identifies (absolute paths, hosts,
  secret-shaped strings, caller-supplied known canaries; fail-closed when
  secrets appear outside redactable free-text fields), normalizes
  compiler/tool state into replayable environment pins, clusters recurring
  failures by a fingerprint of `(operation, error_type, normalized message)`,
  and proposes capability / task / environment / verifier artifacts.
- **Review (`mark_reviewed`)** requires an explicit `reviewer_id` and a full
  human/domain checklist, including separate agent-trajectory and
  verifier-trajectory review items.
- **Freeze (`freeze_candidate`)** requires environment pinning with
  deterministic replay, verifier validation on held positive/negative samples
  (a trivial verifier that accepts its negative or equals its positive is
  rejected), train/eval root-family dedup (no overlap with training families;
  the family must not hash into the train split under
  `RootFamilySplitPolicyV1`), and exposed-answer / shortcut checks. It writes
  an immutable sha256-rowed tamper-evident manifest, create-once;
  `verify_frozen_eval` is fail-closed.
- **Confirmation (`record_confirmation_use`)** is one-touch only; a second
  touch raises (append-only ledger).
- **Promotion (`promote_to_training`)** creates a **new** derivation activity
  and dataset version under a new train-split root family derived from the
  frozen family; it never mutates the frozen eval (byte-identical manifest
  before/after, tested).

## Stop rule

A trace that cannot be safely de-identified (`deidentification_failed`),
reproduced (`environment_not_reproducible` — required pins missing), or
independently verified (`not_independently_verifiable` — no expected output)
stays `QUARANTINED` with coded `rejection_reasons` and cannot be reviewed
(tested for all three).

## Acceptance mapping

| Acceptance criterion | Mechanism |
| --- | --- |
| No trace becomes a frozen eval or training row automatically | ingest always `QUARANTINED`; all transitions explicit |
| Frozen eval root families immutable and disjoint from training | create-once freeze; training-family dedup + split-policy check; tamper-evident sha256 manifest |
| Environment and verifier replay deterministically | pinned tool/compiler state; `verify_environment_replay`; exact-match verifier validated on held samples |
| Privacy, leakage, shortcut, reviewer evidence explicit | redaction report, dedup checks, exposed-answer/shortcut checks, reviewer id + checklist recorded in the frozen manifest |

## Recipe / honesty

- Device: cpu; steps: 0; backend: fixture; matrix set: `slm390_trace_evals`; suite `n`: 0.
- Ship gates: not applicable — this is wiring for the quarantine/freeze
  firewall, not an eval scoreboard. Fixture-demo, not production ship.

## Verification

- `pytest tests/test_harnesses/experiments/test_slm390_trace_evals.py` — 20 passed.
- `python -m scripts.verify_version_stamps --check` — registry component
  `harness.experiments.slm390_trace_evals` v1.
