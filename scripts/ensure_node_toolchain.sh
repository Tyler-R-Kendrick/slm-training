#!/usr/bin/env bash
# Idempotent Node toolchain bootstrap for a fresh checkout/session.
#
# AgentV eval publishing (src/slm_training/evals/agentv.py) and the OpenUI
# language bridge (src/apps/openui_bridge) both require `npm ci` in their
# respective directories before the first eval or DSL-parse run; node_modules
# is gitignored so every fresh checkout starts without it. This has been
# independently rediscovered and documented across many prior
# docs/design/*.md runs (see docs/design/agentv-evaluation.md). Some sandboxes
# also export an ambient NODE_OPTIONS="--import tsx" that breaks plain `npm
# ci`, so it is unset here.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ensure_installed() {
  local dir="$1" sentinel="$2"
  if [ -e "$dir/$sentinel" ]; then
    return 0
  fi
  if [ ! -f "$dir/package.json" ]; then
    return 0
  fi
  (cd "$dir" && env -u NODE_OPTIONS npm ci --silent)
}

ensure_installed "$REPO_ROOT" "node_modules/@agentv/core/package.json"
ensure_installed "$REPO_ROOT/src/apps/openui_bridge" "node_modules/.package-lock.json"
