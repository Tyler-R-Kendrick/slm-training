#!/usr/bin/env bash
# report_merge_readiness.sh — one-screen merge-readiness summary. READ ONLY.
#
# Thin wrapper over `python -m scripts.verify_merge_ready --json` for agents'
# status checks. Mutates nothing (the gate itself only runs read-only
# certificates and the changed-test fan-out). Extra args pass through, e.g.:
#     scripts/report_merge_readiness.sh --fast
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
PY=python3
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
fi

json=$("$PY" -m scripts.verify_merge_ready --json "$@" 2>/dev/null) || true
if [ -z "$json" ]; then
  echo "merge-readiness: gate emitted no JSON (crashed before summary)" >&2
  echo "re-run directly: $PY -m scripts.verify_merge_ready" >&2
  exit 1
fi

MERGE_READY_JSON="$json" "$PY" - <<'EOF'
import json
import os
import sys

summary = json.loads(os.environ["MERGE_READY_JSON"])
verdict = "READY" if summary["ok"] else "BLOCKED"
print(f"# Merge readiness: {verdict}")
print(f"mirrors: {summary['mirrors']}")
print(f"rule:    {summary['rule']}")
print()
print(f"{'step':<24} {'status':<8} {'seconds':>8}")
for step in summary["steps"]:
    print(f"{step['name']:<24} {step['status']:<8} {step['seconds']:>8}")
failed = [s for s in summary["steps"] if s["status"] not in ("ok", "skipped")]
for step in failed:
    print()
    print(f"## {step['name']} ({step['status']}): {step['cmd']}")
    tail = step.get("output_tail", "").strip().splitlines()[-12:]
    print("\n".join(tail))
sys.exit(0 if summary["ok"] else 1)
EOF
