# Autotrain cycle c1 — generate_batch_size schema failure

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Cycle | 1 |
| Campaign | `continuous-loop-20260804-continuous-openui-local-8c0b60dd-c1` |
| Integration | `bf31eb7a3` |
| Status | harness failure (pre-train) |
| Positive | no |

## What happened

Restarted continuous mode after `continuous-openui-20260730` user-stop at cycle 2478.
First supervised cycle on current main failed during matrix validation:

`generate_batch_size` is injected for screening (`run_autotrain_continuous._matrix`,
from #1433 / #1408) so tiny smoke suites get per-record fair-share decode timeouts.
`ExperimentKnobs` still had `extra="forbid"` without that field, so
`HypothesisMatrix` rejected all arms.

## Repair

- Add `generate_batch_size` to `ExperimentKnobs` + `DEFAULT_ALLOWED_KNOBS`
- Forward knob to `scripts.evaluate_model --generate-batch-size` in `engine.compile_commands`
- Regression: screening matrix validates with `generate_batch_size=1`; eval CLI routing test
- Bump `harness.autoresearch.experiment_campaign` **v180 → v181**

## Honesty

Fixture/smoke only. No ship claim. Failure is harness, not model.
