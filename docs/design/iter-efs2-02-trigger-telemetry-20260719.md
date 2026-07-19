# EFS2-02 — Observe-only trigger telemetry before recovery (2026-07-19)

Fixture-grade instrumentation requested by SLM-112. Machine-readable evidence:
[`iter-efs2-02-trigger-telemetry-20260719.json`](iter-efs2-02-trigger-telemetry-20260719.json).
Linear SLM-112.

## What ran

The new `slm_training.evals.search_trigger_telemetry` harness was exercised by
`scripts/run_efs2_02_trigger_telemetry_fixture.py` over four synthetic
decision trajectories and three decode regimes:

- **greedy** compiler-tree control;
- **temperature** sampling with small additive score noise;
- **beam** search with small random tie-breaks.

The observer is strictly observe-only: it records `SearchTriggerObservationV1`
events but never branches, remasks, injects noise, or changes tie-breaking.
Regression tests assert byte-identical observations across runs, no input
mutation, bottom-as-retraction, and stagnation/uncertainty/budget triggers.

## Threshold manifest (frozen)

```json
{
  "repeat_window": 3,
  "no_progress_window": 4,
  "margin_quantile": 0.1,
  "entropy_quantile": 0.75,
  "value_plateau_window": 3,
  "budget_pressure_forward_limit": 5
}
```

## Trigger firing summary

| Trajectory | Regime | Observations | Fired | Rate | Predicates |
| --- | --- | ---: | ---: | ---: | --- |
| stagnation_path | greedy | 5 | 3 | 0.60 | STAGNATION 3, UNCERTAINTY 2 |
| bottom_path | greedy | 2 | 1 | 0.50 | BOTTOM 1, UNCERTAINTY 1 |
| uncertainty_path | greedy | 4 | 4 | 1.00 | UNCERTAINTY 4 |
| budget_path | greedy | 8 | 3 | 0.375 | BUDGET_PRESSURE 3, UNCERTAINTY 5 |
| stagnation_path | temperature | 5 | 3 | 0.60 | STAGNATION 3, UNCERTAINTY 2 |
| bottom_path | temperature | 2 | 1 | 0.50 | BOTTOM 1, UNCERTAINTY 1 |
| uncertainty_path | temperature | 4 | 4 | 1.00 | UNCERTAINTY 4 |
| budget_path | temperature | 8 | 3 | 0.375 | BUDGET_PRESSURE 3, UNCERTAINTY 5 |
| stagnation_path | beam | 5 | 3 | 0.60 | STAGNATION 3, UNCERTAINTY 2 |
| bottom_path | beam | 2 | 1 | 0.50 | BOTTOM 1, UNCERTAINTY 1 |
| uncertainty_path | beam | 4 | 4 | 1.00 | UNCERTAINTY 4 |
| budget_path | beam | 8 | 3 | 0.375 | BUDGET_PRESSURE 3, UNCERTAINTY 5 |

Mean firing rate is identical across regimes (0.619) because the fixture
replays the same synthetic decision sequences for every regime; only the score
perturbation differs between temperature and beam.

## Observations against the falsifier

1. **The observer emits all four requested predicates**: `BOTTOM` is recorded as
   a retraction event on the hard-conflict trajectory, `STAGNATION` fires on
   repeated state fingerprints, `UNCERTAINTY` fires on low-margin/high-entropy
   action sets, and `BUDGET_PRESSURE` fires when `model_forwards` exceeds the
   declared limit.
2. **The observer does not alter caller state**: tests assert the input
   `DecisionStep` is unchanged and that repeated runs with the same seed produce
   byte-identical observation dictionaries.
3. **No Phase B gate is passed**: the fixture uses hand-designed synthetic
   sequences, not real checkpoints or binding-aware outcome labels, so the
   preregistered activation gate for enabled recovery is **not** claimed.

## Verdict

`diagnostic_only`

The trigger telemetry machinery is retained as an observe-only diagnostic
fixture. Phase B (triggered deterministic branch, beam expansion, or PTRM-style
recovery) is **not** enabled because this iteration provides only wiring
evidence on synthetic sequences. A later issue must demonstrate nonzero,
nontrivial firing rate with useful precision/recall on durable checkpoints and
frozen suites before any recovery action can be gated.

## Artifacts

- `src/slm_training/evals/search_trigger_telemetry.py` — schema/collector and regime comparator.
- `tests/test_evals/test_search_trigger_telemetry.py` — regression tests.
- `scripts/run_efs2_02_trigger_telemetry_fixture.py` — wiring fixture CLI.
- `outputs/runs/efs2-02-trigger-telemetry/iter-efs2-02-20260719/trigger_telemetry.json` — full raw traces.
- `outputs/runs/efs2-02-trigger-telemetry/iter-efs2-02-20260719/summary.json` — aggregate summary.
- `docs/design/iter-efs2-02-trigger-telemetry-20260719.json` — durable machine-readable evidence.

## Honesty and limits

- **Wiring evidence only, not a ship claim.** No checkpoint is loaded, no model
  is run, and the decision sequences are synthetic.
- **Thresholds are frozen** for this fixture but are *not* derived from a
  validation split; they are hand-set defaults. A real Phase A run must select
  quantiles from a validation set and freeze them before test analysis.
- **Outcome labels are post-hoc and uniform per trajectory** in this fixture. A
  real implementation needs per-step offline-oracle labels tied to
  binding-aware meaningful pass/fail and recoverability.
- **Default decode is unchanged.** No runtime recovery, remasking, or PTRM
  trajectory ordering is enabled.
- Component `evals.scoring` was bumped to `v2` to reflect the new harness.
