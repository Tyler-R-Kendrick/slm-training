# Autotrain c1745: `_apply_frozen_replay` multi-hyphen slug repair

**Verdict:** canonical harness defect, repaired. The c4→c5
`retry_measurement` of a `component-edge` screening arm crashed the
continuous-loop driver instead of replaying, blocking every automated
frozen-replay retry for 5 of the 10 `_SCREENING_ARM_BANK` entries. This is a
`repair_harness` action, not a model result.

## Reproduction

```
CYCLE_ERROR RuntimeError('unsupported automatic frozen replay arm: edge')
```

`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c4`'s candidate
arm was `c20260802-continuous-openui-local-8c0b60dd-c4-component-edge`
(bank slug `component-edge`). The successor cycle's
`scripts.run_autotrain_continuous._apply_frozen_replay` recovered the slug
with `old_candidate_id.rsplit("-", 1)[-1]`, which only returns the text
after the *last* hyphen — `"edge"` instead of `"component-edge"` — so the
membership check against `_SCREENING_ARM_BANK` always failed for any
multi-word bank slug (`component-plan`, `component-edge`,
`component-inventory`, `binder-topology`, `component-structure`).

## Repair

`scripts/run_autotrain_continuous.py::_apply_frozen_replay` now recovers the
slug by matching the frozen candidate's experiment id against the known
`_SCREENING_ARM_BANK` slugs (longest-first suffix match:
`old_candidate_id.endswith(f"-{slug}")`) instead of a naive single-hyphen
`rsplit`. Added `test_apply_frozen_replay_recovers_multi_hyphen_slug` and
`test_apply_frozen_replay_rejects_unknown_arm` to
`tests/test_scripts/test_run_autotrain_continuous.py`
(`pytest tests/test_scripts/test_run_autotrain_continuous.py`: 110 passed,
1 skipped). Bumped `harness.autoresearch.experiment_campaign` v60 → v61.

## Next-run priorities

1. Replay the identical frozen `component-edge` arm from campaign
   `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c4` now that
   the slug-recovery repair is landed; require a complete `held_out`
   scoreboard before any promotion disposition.
2. No model claim is made by this fix; it only restores the harness's
   ability to replay 5 previously-unreplayable screening arms.

No checkpoint was promoted; `docs/MODEL_CARD.md` / README are unchanged.
