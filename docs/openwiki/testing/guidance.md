# Testing guidance

## Local hooks

Configure the tracked hooks with `git config core.hooksPath .githooks`.
Pre-commit runs the staged changed-file check. Pre-push runs:

- changed-test selection against the outgoing commits
- repository, decode, agent-surface, checkpoint, tokenizer/grammar, and page-DSL certificates
- `ruff check .`
- `python -m compileall -q src scripts tests`

Install: `pip install -e ".[dev,mcp,web]"`.

Vendored skill trees under `.agents/skills/` (HF marketplace, headroom helpers, caveman scripts) are excluded from ruff via `pyproject.toml`.

## Ship honesty

Use skill `honest-ship-eval` before claiming readiness:

- Multi-suite `--ship-gates`
- Full held-out / HF / DESIGN.md when claimed
- Do not equate scratch matrix wins with production ship

Adversarial inventory: [`docs/design/adversarial-review.md`](../../design/adversarial-review.md).

## Standard eval format

Model, loss, task, and diagnostic eval runs publish AgentEvals JSONL plus an
AgentV SDK bundle. Install the exact npm dependencies with `npm ci`. Missing
model suites remain failed AgentV cases; AgentV never relaxes the canonical
OpenUI ship gates. See
[`docs/design/agentv-evaluation.md`](../../design/agentv-evaluation.md).

## Structure-only eval

See [`docs/design/structure-only-eval.md`](../../design/structure-only-eval.md) and `scripts/evaluate_model.py` / `evaluate_loss_suites.py`.

## Playground e2e

Skill `playwright-cli` for browser / playground automation.
