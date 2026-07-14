# Testing guidance

## Local / CI

Default CI job (`.github/workflows/ci.yml`) runs:

- `ruff check .`
- `python -m compileall`
- `pytest -q`
- Documented disjoint data-build check

Install: `pip install -e ".[dev,mcp,web]"`.

Vendored skill trees under `.agents/skills/` (HF marketplace, headroom helpers, caveman scripts) are excluded from ruff via `pyproject.toml`.

## Ship honesty

Use skill `honest-ship-eval` before claiming readiness:

- Multi-suite `--ship-gates`
- Full held-out / HF / DESIGN.md when claimed
- Do not equate scratch matrix wins with production ship

Adversarial inventory: [`docs/design/adversarial-review.md`](../docs/design/adversarial-review.md).

## Structure-only eval

See [`docs/design/structure-only-eval.md`](../docs/design/structure-only-eval.md) and `scripts/evaluate_model.py` / `evaluate_loss_suites.py`.

## Playground e2e

Skill `playwright-cli` for browser / playground automation.
