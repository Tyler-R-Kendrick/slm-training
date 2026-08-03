# Autotrain c1 (continuous-openui-local): torch missing, not a model result

**Verdict:** infrastructure failure, not scoreable. This session's sandbox
`.venv` had `slm-training` installed with `--no-deps`, so both the control and
`-bounds` `wf_smoke_v2` arms crashed inside `scripts/train_model.py` at
`detect_device()` with a bare `ModuleNotFoundError: No module named 'torch'`
before any step ran. Neither arm produced a checkpoint, a scoreboard, or smoke
metrics; this is not evidence about the model, the compiler-tree grammar
objective, or any lever.

The continuous driver's `SDLC_PHASE_A` classification flagged this correctly
as `NON_POSITIVE` / `harness_failure`, and its typed handoff routed a
`repair_harness` action against `model_build` before permitting any new model
hypothesis.

Root cause and fix (commit `1faeff44`): `detect_device()` did a bare
`import torch` with no fallback, so a missing dependency surfaced as an
opaque traceback deep in the training script instead of a clear, actionable
message. Fixed:

- `detect_device()` now raises `RuntimeError("torch is not installed in this
  environment. Run scripts/setup_dev_env.sh ...")` on `ModuleNotFoundError`.
- Added `scripts/setup_dev_env.sh`, a CI-parity bootstrap (Python 3.12 venv +
  pinned `torch==2.5.1+cpu` wheel from the PyTorch CPU index) so future local
  or agent sessions in this repo do not hit the same gap.
- Regression test:
  `tests/test_runtime/accel/test_accel.py::test_detect_device_missing_torch_raises_actionable_error`
  asserts the new message on a simulated missing-torch import.

No checkpoint was created; nothing here is reusable, promotable, or ship
evidence. Lean is `not_applicable:screening`. AgentV bundles never ran to
completion (both arms failed before decode).

Next: replay the identical frozen `-bounds` arm (`retry_measurement`) now that
torch is installed in this environment and the harness fails closed with an
actionable message instead of an opaque crash.

Machine evidence:
[`autotrain-cycle-c1-torch-missing-infra-failure.json`](autotrain-cycle-c1-torch-missing-infra-failure.json).
