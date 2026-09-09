# SLM-587 / A11: fail-closed record loading

The model-build loader previously checked symbol-only targets but omitted the
trainer's opaque-marker, semantic-label and placeholder-role checks. It could
therefore admit records that `TwoTowerModel.from_records` immediately refuses.
The loader now calls `data.record_admission.assert_training_record`, preserving
the existing four trainer contracts and Harness DSL metadata validation.
No input is normalized, dropped or rewritten to pass admission.

Local verification on baseline `2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`
plus this patch: **26 passed in 2.42 seconds**, exit 0. Six new cases cover
invalid RadialChart roles, semantic labels in prompt/DESIGN.md, named markers,
and valid document/lexical targets. Rejection is compared with the actual
trainer constructor and certified admission owner; persisted bytes are unchanged.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -o addopts= -m '' -q tests/test_harnesses/model_build/test_record_admission.py tests/test_harnesses/model_build/test_harness_data.py tests/test_harnesses/test_data/test_certified_admission_mutation.py
```

Run through the canonical 170-second interrupt plus 10-second kill grace.
`repo_policy`, `verify_version_stamps --check`, `extract_test_cases --write`
for the new test, and `git diff --check` all exited 0. The loader component
advances from v2 to v3; old evaluation records are not relabeled or remeasured.

Delivery is based on remote `a809e81878302baa27a9d6fedd8f0b79ee3af857`,
preserving intervening changes. These are focused contract tests, not full-tree
release authorization, training evidence or a live autonomous repair demonstration.
The broader A11 readiness/publication/dispatch closure remains separate: remote
data-rebuild imports readiness modules not yet present in that tree. This PR
does not claim to restore that entire missing package or install/start a service.
