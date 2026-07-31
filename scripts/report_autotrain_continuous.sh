#!/usr/bin/env bash
# autotrain-report.sh — user-facing liveness + skill-required matrix. READ ONLY.
set -euo pipefail
WT=${AUTOTRAIN_WT:-/tmp/slm-autotrain-continuous-loop}
LOOP_ID=${AUTOTRAIN_LOOP_ID:-continuous-openui-20260730}
ROOT=${AUTOTRAIN_ROOT:-$WT/outputs/autoresearch}
OUT_STATUS=${AUTOTRAIN_STATUS:-/tmp/autotrain-loop-status.txt}
OUT_MATRIX=${AUTOTRAIN_MATRIX:-/tmp/autotrain-loop-matrix.md}
OUT_DASH=${AUTOTRAIN_DASH:-/tmp/autotrain-loop-dashboard.md}
PY=${AUTOTRAIN_PY:-$WT/.venv/bin/python}

driver_pid=$(ps -eo pid,cmd | awk -v id="$LOOP_ID" '
  $0 ~ /scripts.run_autotrain_continuous/ && $0 ~ id && $0 !~ /awk/ {print $1; exit}
')
now=$(date -Is)
load=$(cut -d' ' -f1-3 /proc/loadavg)
latest=$(ls -1dt "$ROOT"/continuous-loop-* 2>/dev/null | head -1 | xargs -r basename || true)
child=$(ps -eo pid,pcpu,etime,cmd --sort=-pcpu | awk '
  /slm-autotrain-continuous-loop\/.venv\/bin\/python -m scripts\.(train_model|evaluate_model|autoresearch)/ && !/awk/ {
    printf "%s %s%% %s ", $1,$2,$3; for(i=4;i<=NF;i++) printf "%s ",$i; print ""; exit
  }' || true)

if [ -n "${driver_pid:-}" ] && [ -d "/proc/$driver_pid" ]; then
  state=RUNNING
else
  state=DEAD
fi

{
  echo "updated=$now"
  echo "driver_state=$state"
  echo "driver_pid=${driver_pid:-none}"
  echo "load=$load"
  echo "latest_campaign=${latest:-none}"
  echo "top_child=${child:-none}"
  if [ -f "$ROOT/sdlc_delivery_ledger.jsonl" ]; then
    echo "ledger_lines=$(wc -l < "$ROOT/sdlc_delivery_ledger.jsonl")"
  fi
} > "$OUT_STATUS"

# skill-required three-table matrix
if [ -x "$PY" ] && [ -d "$ROOT" ]; then
  "$PY" -m scripts.autoresearch --root "$ROOT" status --loop-id "$LOOP_ID" --matrix --last 5 \
    > "$OUT_MATRIX" 2>&1 || echo "(matrix unavailable)" > "$OUT_MATRIX"
else
  echo "(python or root missing)" > "$OUT_MATRIX"
fi

{
  echo "# Autotrain continuous dashboard"
  echo
  echo "Updated: \`$now\`"
  echo
  echo "## Liveness (do not use Grok background UI for this)"
  echo
  echo "| Field | Value |"
  echo "| --- | --- |"
  echo "| driver_state | **$state** |"
  echo "| driver_pid | \`${driver_pid:-none}\` |"
  echo "| load | \`$load\` |"
  echo "| latest_campaign | \`${latest:-none}\` |"
  echo "| top_child | \`${child:-none}\` |"
  echo
  echo "Status file: \`$OUT_STATUS\`  "
  echo "Matrix file: \`$OUT_MATRIX\`  "
  echo "Log: \`/tmp/autotrain-continuous-driver.log\`"
  echo
  echo "## Skill matrix (last 5 cycles)"
  echo
  cat "$OUT_MATRIX"
} > "$OUT_DASH"

# also print to stdout for agents to paste
cat "$OUT_DASH"
