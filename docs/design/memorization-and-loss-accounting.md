# SLM-261: Bounded memorization and loss accounting

This document is the durable home for the VSD0-02 trainer-validity probe.
Per-run measured results live in the dated iter files:

- `docs/design/iter-slm261-memorization-probe-20260721.json`
- `docs/design/iter-slm261-memorization-probe-20260721.md`

## Goal

Demonstrate that the canonical TwoTower training path can memorize a small,
verified semantic corpus and that every reported loss is interpretable.  This
is a **trainer-validity** experiment, not a generalization or ship run.

## What changed

- `src/slm_training/models/loss_ledger.py` — typed `LossLedgerV1` dataclass
  that reconstructs every active loss term (raw value, coefficient, weighted
  contribution) from the flat `last_training_metrics` dict and fails closed if
  the reconstructed total does not match the reported total.
- `src/slm_training/models/twotower.py` — `training_loss` now emits raw,
  weight, and contribution metrics for every auxiliary term and embeds a
  `loss_ledger` entry in `last_training_metrics` every step.
- `src/slm_training/harnesses/experiments/slm261_memorization_probe.py` —
  fixture harness that builds a deterministic `MemorizationCorruptionSuiteV1`,
  runs M0 (principal-only) and M1 (current recipe) arms, and reports fixed-
  corruption NLL, exact target accuracy, loss-ledger reconciliation, and
  trainable-parameter counts.
- `scripts/run_memorization_probe.py` — canonical CLI.

## CLI

```bash
python -m scripts.run_memorization_probe \
  --corpus src/slm_training/resources/data/eval/remediated/suites/smoke/records.jsonl \
  --n-records 3 --steps 3 --fast \
  --output-dir outputs/experiments/slm261_memorization_probe \
  --write-design-docs
```

## Arms

- **M0 — principal-only:** every auxiliary weight forced to zero; masked
  denoising CE only.
- **M1 — current recipe:** the existing production/research recipe with every
  active loss and coefficient recorded in the ledger.
- **M2 — candidate-normalized CE:** deferred; requires the VSD0-01 scorer and
  legal-candidate plumbing to be clean first.

## Honest caveats

- Fixture-only diagnostic: tiny model, tiny corpus, CPU run.
- VSD0-01 semantic scorer prerequisite is not enforced by this fixture.
- M2 is not implemented in this iteration.
- Exact canonical reconstruction is measured by string/canonical match, not the
  full binding-aware meaning pipeline.
- A full 50-record `memorization50_v1` corpus is pending VSD0-01 clean scoring;
  the harness is intentionally generic so it can consume that corpus once ready.

## Disposition semantics

The fixture publishes one of the VSD0-02 labels based on the tiny corpus:

- `trainer_memorizes` — an arm reaches exact target accuracy ≥ 0.99 and loss-
  ledger reconciliation error < 1e-3.
- `inconclusive` — the tiny fixture does not reach the strict thresholds; this
  is expected and is **not** a falsification of trainer validity.
- Future iterations may add `auxiliary_stack_blocks_memorization`,
  `principal_objective_untrainable`, or `label_or_optimizer_defect_fixed` once
  the 50-record corpus and VSD0-01 scorer are in place.
