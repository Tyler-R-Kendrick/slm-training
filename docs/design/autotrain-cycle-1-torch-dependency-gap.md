# Autotrain c1: fresh-container `torch` dependency gap

**Verdict:** infrastructure failure only — no model measurement. This is
cycle 1 of a new local loop (`continuous-openui-20260802-local`) started in a
fresh, ephemeral container (`slm-training` editable-installed via
`pip install -e .` with no prior venv). Both arms failed identically before
any training step ran, so there is no arm comparison, no primary-metric
value, and no promotion/ship claim from this cycle.

## Result matrix

| Arm | Command | Result | Disposition |
| --- | --- | --- | --- |
| control | `train_model --train-version wf_smoke_v2 --device cpu ...` | exit 2: `ModuleNotFoundError: No module named 'torch'` | Infra repair required; not scoreable |
| bounds | same, `--grammar-completion-bounds` | exit 2: identical `ModuleNotFoundError` | Infra repair required; not scoreable |

Full command lines and tracebacks are in the
[machine-readable record](autotrain-cycle-1-torch-dependency-gap.json).

## Diagnosis

`src/slm_training/runtime/accel/__init__.py:28` imports `torch` inside
`detect_device`, which every `train_model` invocation calls before touching
any recipe knob. `torch` is declared as the optional `torch` extra in
`pyproject.toml` (`torch = ["torch>=2.2,<2.6"]`), not a base dependency, so a
bare `pip install -e .` in a clean container leaves it absent. This is an
environment/dependency gap, not a harness or model regression — both arms
failed identically regardless of the `grammar_completion_bounds` knob under
test.

## Repair applied

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu
```

Verified with `python -c "import torch; print(torch.__version__)"` →
`2.5.1+cpu`. Cycle 2 replays the identical control/bounds specs with the
repaired venv. Follow-up containers in this loop should prefer
`pip install -e '.[torch]'` (or an equivalent extras-aware install) over the
bare editable install to avoid re-hitting this gap.

No harness code changed, so no `versions.json` component bump is required —
this is an environment setup gap, not a canonical-harness defect.

Eval commit: `cb3f557663bed01d9a35f304ff65b8f8a19eefcc`
(`model.twotower=v276`).
