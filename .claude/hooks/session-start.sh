#!/usr/bin/env bash
# Idempotent dev bootstrap for fresh Claude Code remote/web containers.
# Skips local workstations unless CLAUDE_CODE_REMOTE is set.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

repo_root="$(git -C "${CLAUDE_PROJECT_DIR:-.}" rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$repo_root"

if [ ! -d .venv ]; then
  python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate

python -m pip install -q --no-deps -e .
python -m pip install -q \
  "pytest>=8,<9" "pytest-asyncio>=0.23,<2" "ruff>=0.9,<0.16" \
  "numpy>=1.26,<3" "httpx>=0.27,<1" "fastapi>=0.110,<1" \
  "lark>=1.1,<2" "openfeature-sdk>=0.10,<1" "pydantic>=2.7,<3" \
  "PyYAML>=6,<7" "onnxruntime>=1.18,<2"
python -m pip install -q --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"

if [ -f package-lock.json ] && [ ! -f node_modules/.package-lock.json ]; then
  env -u NODE_OPTIONS npm ci --silent
fi

bridge_dir="src/apps/openui_bridge"
if [ -f "$bridge_dir/package-lock.json" ] && [ ! -f "$bridge_dir/node_modules/.package-lock.json" ]; then
  (cd "$bridge_dir" && env -u NODE_OPTIONS npm ci --silent)
fi
