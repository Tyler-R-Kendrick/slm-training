#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-.}"

# Python: repo requires >=3.12,<3.13; bootstrap an isolated venv so
# `python -m scripts.*` and the `slm_training` package resolve.
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 .venv >/dev/null 2>&1 || python3.12 -m venv .venv
  else
    python3.12 -m venv .venv
  fi
fi
if ! .venv/bin/python -c "import slm_training" >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -e ".[dev]" >/dev/null 2>&1
  else
    .venv/bin/python -m pip install -e ".[dev]" >/dev/null 2>&1
  fi
fi

# Node: the AgentV-gated eval path (src/slm_training/evals/agentv.py) hard
# requires node_modules/@agentv/core. The sandbox's ambient NODE_OPTIONS is
# malformed for plain `npm ci`/`npm install`, so sanitize it for this call
# only rather than mutating the global env.
if [ ! -f node_modules/@agentv/core/package.json ]; then
  NODE_OPTIONS="--max-old-space-size=8192" npm install >/dev/null 2>&1 || true
fi

exit 0
