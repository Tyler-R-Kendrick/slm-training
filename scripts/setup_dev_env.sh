#!/usr/bin/env bash
# Bootstrap a local .venv matching CI's Python 3.12 + pinned CPU torch setup.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --no-deps -e .
python -m pip install \
  "pytest>=8,<9" "pytest-asyncio>=0.23,<2" "ruff>=0.9,<0.16" \
  "numpy>=1.26,<3" "httpx>=0.27,<1" "fastapi>=0.110,<1" \
  "lark>=1.1,<2" "openfeature-sdk>=0.10,<1" "pydantic>=2.7,<3" \
  "PyYAML>=6,<7" "onnxruntime>=1.18,<2"
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"

# AgentV SDK for evaluate_model.py --ship-gates (scripts/run_agentv_eval.mjs).
env -u NODE_OPTIONS npm ci

echo "Ready: source .venv/bin/activate"
