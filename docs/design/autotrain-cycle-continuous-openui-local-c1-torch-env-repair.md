# Autotrain continuous-openui-local c1: fresh-container torch bootstrap gap (infrastructure, not a model result)

**Verdict:** cycle 1 of loop `continuous-openui-local` (campaign
`continuous-loop-20260802-continuous-openui-local-8c0b60dd-c1`) is an
infrastructure failure, not a quality-null model result. Both the control and
bounds arms failed identically inside `scripts/train_model.py`'s
`detect_device` with `ModuleNotFoundError: No module named 'torch'` before any
training step ran. `measurement_incomplete` on both arms; no metrics, no
checkpoints, no ship gate evidence.

**Root cause:** this scheduled session's Python 3.12 venv was created with a
plain `pip install -e .`, which only pulls the base `pyproject.toml`
dependency set. `torch` is declared as an optional extra
(`[project.optional-dependencies].torch` / `.dev`), required for TwoTower
training but not installed by default — an ephemeral-container bootstrap gap,
not a harness code defect.

**Fix:** reinstalled with `pip install -e ".[torch,dev]"`, resolving torch
2.5.1+cu124. No repository code changed. This matches the same class of
first-run environment gap documented in prior sessions (e.g.
`autotrain-cycle-1pse54-c1-torch-env-repair.md`, and the earlier
`continuous-openui-local` c1 bounds screen recorded in this loop's history).

**Next step:** per the driver's own priority queue, replay the identical
frozen control/candidate arm (`retry_measurement`) with the repaired
environment before evaluating any new hypothesis — no knob changes, same
`train_version=wf_smoke_v2`, `steps=20`, seed, and manifest.

Lean is `not_applicable:no_champion`; no ship or climb claim is made here.

Machine evidence:
[`autotrain-cycle-continuous-openui-local-c1-torch-env-repair.json`](autotrain-cycle-continuous-openui-local-c1-torch-env-repair.json).
