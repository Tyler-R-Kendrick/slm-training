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
if [ "$(uname -s)-$(uname -m)" = "Linux-x86_64" ]; then
  python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.5.1+cpu"
else
  python -m pip install "torch==2.5.1"
fi

# Nix Python does not search the host C++ runtime directory. Keep its libc
# isolated and expose only libstdc++ beside Torch's shared libraries.
if ! python -c 'import torch'; then
  libstdcpp="$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6/{print $NF; exit}')"
  torch_lib="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("torch").submodule_search_locations)))')/lib"
  if [ -f "$libstdcpp" ] && [ ! -e "$torch_lib/libstdc++.so.6" ]; then
    ln -s "$libstdcpp" "$torch_lib/libstdc++.so.6"
  fi
  python -c 'import torch'
fi

# AgentV SDK for evaluate_model.py --ship-gates (scripts/run_agentv_eval.mjs).
env -u NODE_OPTIONS npm ci

# OpenUI bridge for G2/G8 whole-program verification (binding_aware_meaningful_v2).
if [ -f src/apps/openui_bridge/package-lock.json ]; then
  (cd src/apps/openui_bridge && env -u NODE_OPTIONS npm ci)
fi

echo "Ready: source .venv/bin/activate"
