# Continuous autotrain cycle 1 results (2026-08-01, loop `continuous-openui-bqw0tb`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-bqw0tb` |
| Campaign | `continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c1` |
| Source | `b5134b90532986384fbd34c9fd609b2681bfe390` |
| Device | CPU |
| Steps | 21 / batch 2 / seed 100001 |
| Train | `wf_smoke_v2` |
| Eval | `e938_role_safe_all_targets_v2` |
| Wall cap | 3 minutes |

## Run matrix

| Arm | Levers | Status |
| --- | --- | --- |
| c1-control | bounds off | **failed** — `ModuleNotFoundError: No module named 'torch'` in `scripts.train_model` → `detect_device` |
| c1-bounds | bounds **on** | **failed** — identical error; train stage never started |

No metrics were produced by either arm.

## Diagnostics

1. This scheduled continuous-loop session started from a fresh container with
   no pre-existing Python environment for `slm_training`. A `python3.12 -m
   venv` was created and the editable package plus the pinned lightweight
   dependency set (pytest, ruff, numpy, httpx, fastapi, lark,
   openfeature-sdk, pydantic, PyYAML — the same set CI installs
   unconditionally) were installed, but `torch` was not installed alongside
   them.
2. `scripts/train_model.py`'s `detect_device` imports `torch` unconditionally,
   so both the control and bounds arms failed identically at the very start
   of the train stage — before any lever, model, or data code ran. This is
   an environment-setup gap, not a model or lever finding.
3. Fix applied immediately after diagnosis: `pip install --index-url
   https://download.pytorch.org/whl/cpu torch==2.5.1+cpu` into the same venv
   (mirroring the CPU wheel `.github/workflows/ci.yml` installs for CI). The
   driver's own diagnosis recorded a `retry_measurement` action against the
   frozen arm manifests, which the successor cycle consumes to replay the
   identical `c1-control` / `c1-bounds` specs now that torch is present.

## Next-run priorities

1. **infrastructure:** replay the frozen `c1-control` / `c1-bounds` arms
   (torch now installed) before starting any new model hypothesis.
2. **infrastructure:** consider a documented one-shot venv bootstrap step for
   fresh continuous-loop containers so future sessions don't lose a cycle to
   the same gap.
3. **evaluation:** soft ship-gate/infra fails on fixture `n` never stop the
   continuous loop; proceed to the next cycle regardless.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260801-continuous-openui-bqw0tb-f852ac38-c1/`
- JSON twin: `continuous-openui-bqw0tb-c1-results.json`
