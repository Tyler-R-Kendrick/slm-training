# EFS0-05 — Rejected-lever registry and five-seed paired re-adjudication (2026-07-19)

Fixture-grade implementation requested by SLM-107. Machine-readable evidence:
[`iter-efs0-05-rejected-lever-readjudication-20260719.json`](iter-efs0-05-rejected-lever-readjudication-20260719.json).
Linear SLM-107.

## What ran

The new `slm_training.harnesses.experiments.rejected_lever_registry` module was
exercised by `scripts/run_efs0_05_rejected_lever_fixture.py` with a synthetic
registry of five historically rejected levers and a four-row preregistered
campaign (one lever is already marked `closed` and is retained as a sentinel).

### Registry contents

| Entry | Matrix | Original metric | Confounds | Status |
| --- | --- | --- | --- | --- |
| E175 retrieval | quality | `binding_aware_meaningful_v2` | `representation_mismatch`, `underexposure` | `reopen_candidate` |
| E255/E256 AR→diffusion | quality | `binding_aware_meaningful_v2` | `tiny_n`, `underexposure` | `reopen_candidate` |
| X9/X14 typed topology | grammar | `binding_aware_meaningful_v2` | `tiny_n`, `seed_instability` | `reopen_candidate` |
| E263 set-valued preference | quality | `binding_aware_meaningful_v2` | `harness_interference`, `tiny_n` | `reopen_candidate` |
| E244 always-on PTRM | quality | `binding_aware_meaningful_v2` | `strong_negative_control`, `decoder_bug` | `closed` (sentinel) |

### Campaign design

- Seeds: `0, 1, 2, 3, 4` for every selected lever.
- Treatment/control pairing is explicit per row (`control_run_id`,
  `treatment_run_id`) and shares the same decoder path
  (`current_exact_or_compiler`), metric, and seed list.
- The fixture generates deterministic synthetic observations tuned to exercise
  the preregistered verdict classes.

### Verdict summary

| Row | Mean Δ | 95% CI | Verdict |
| --- | --- | --- | --- |
| E175 retrieval | +0.065 | [+0.057, +0.073] | `reopened_positive` |
| E255/E256 AR→diffusion | +0.005 | [-0.003, +0.013] | `equivalent` |
| X9/X14 typed topology | -0.015 | [-0.023, -0.007] | `confirmed_negative` |
| E263 set preference | +0.030 | [+0.019, +0.041] | `confirmed_negative`¹ |

¹ CI is below the preregistered `min_effect=0.05` and outside the equivalence
band, so the decision contract classifies it as negative despite a positive
point estimate. This is the intended behavior: point estimates alone cannot
reopen a lever.

The sentinel E244 lever remains `closed`; its signature is included in
`closed_signatures` and is available to autoresearch validation as a branch that
must not be re-proposed.

## Added artifacts

- `src/slm_training/harnesses/experiments/rejected_lever_registry.py` —
  `RejectedLeverV1`, `RejectedLeverRegistryV1`, `ReAdjudicationRowV1`,
  `PairedSeedObservation`, `PairedTestResult`, plus loader, duplicate checks,
  deterministic bootstrap CI, paired classification, and autoresearch
  `EvidenceItem` conversion.
- `scripts/run_efs0_05_rejected_lever_fixture.py` — wiring fixture that builds
  the synthetic registry, runs the campaign, writes the durable JSON, and
  verifies evidence intake.
- `tests/test_harnesses/experiments/test_rejected_lever_registry.py` —
  regression tests for schema validation, duplicate detection, seed
  completeness, failure/timeout retention, deterministic statistics, verdict
  classification, and evidence ingestion.
- `src/slm_training/autoresearch/evidence.py` — classifies files matching
  `rejected_lever` as kind `rejected_lever` and emits a compact summary.
- `docs/design/iter-efs0-05-rejected-lever-readjudication-20260719.md` and
  `.json`.
- `outputs/runs/efs0-05-rejected-lever/iter-efs0-05-20260719/rejected_lever_registry.json`
  and `summary.json`.

## Verdict

`diagnostic_only`

The registry schema, campaign wiring, paired statistical contract, and
autoresearch evidence integration are implemented and tested.  No actual
checkpoints are loaded, no real model runs, and the observations are synthetic.
A production EFS0-05 run must populate the registry from committed evidence,
attach durable checkpoint hashes and remote URIs, execute the levers under the
corrected decoder selected by SLM-104, and ingest independent labels before any
roadmap status is changed.

## Honesty and limits

- **Wiring evidence only, not a ship claim.** The fixture uses hand-designed
  synthetic observations; no checkpoint is loaded and no model runs.
- **Five candidate levers, four campaign rows.** E244 is retained as a closed
  sentinel; the other four are selected for re-adjudication, satisfying the
  4–6 lever bound.
- **Bootstrap CI is a seed-level normal approximation.** It is deterministic and
  adequate for fixture evidence, but a production run should report exact
  paired-test p-values and multiple-comparison correction across selected
  levers.
- **Verdicts are synthetic.** They demonstrate the decision contract, not real
  re-adjudication conclusions.
- Component `harness.experiments` was bumped to `v4` to reflect the new registry
  module.
