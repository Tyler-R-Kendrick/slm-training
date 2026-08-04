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
LOOP_STATE=$ROOT/loops/$LOOP_ID/state.json

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

phase=unknown
next_action=unknown
blocker_count=0
heartbeat=unknown
if [ -f "$LOOP_STATE" ] && command -v jq >/dev/null 2>&1; then
  phase=$(jq -r '.phase // "unknown"' "$LOOP_STATE")
  next_action=$(jq -r '.next_action // "unknown"' "$LOOP_STATE")
  blocker_count=$(jq -r '.blocker_count // 0' "$LOOP_STATE")
  heartbeat=$(jq -r '.heartbeat_at // "unknown"' "$LOOP_STATE")
fi

if [ -n "${driver_pid:-}" ] && [ -d "/proc/$driver_pid" ]; then
  state=RUNNING
elif [ -f "$LOOP_STATE" ] && command -v jq >/dev/null 2>&1; then
  recorded_state=$(jq -r '.state // "DEAD"' "$LOOP_STATE")
  if [ "$recorded_state" = "BLOCKED" ]; then
    state=BLOCKED
  else
    state=DEAD
  fi
else
  state=DEAD
fi

status_tmp=${OUT_STATUS}.tmp.$$
{
  echo "updated=$now"
  echo "driver_state=$state"
  echo "driver_pid=${driver_pid:-none}"
  echo "load=$load"
  echo "latest_campaign=${latest:-none}"
  echo "top_child=${child:-none}"
  echo "phase=$phase"
  echo "next_action=$next_action"
  echo "blocker_count=$blocker_count"
  echo "heartbeat=$heartbeat"
  if [ -f "$ROOT/sdlc_delivery_ledger.jsonl" ]; then
    echo "ledger_lines=$(wc -l < "$ROOT/sdlc_delivery_ledger.jsonl")"
  fi
} > "$status_tmp"
mv "$status_tmp" "$OUT_STATUS"

# Skill-required compact four-table view. Force this worktree's source even
# when its Python executable comes from a shared editable environment.
matrix_tmp=${OUT_MATRIX}.tmp.$$
if [ -x "$PY" ] && [ -d "$ROOT" ]; then
  PYTHONPATH="$WT/src" "$PY" -m scripts.autoresearch --root "$ROOT" status --loop-id "$LOOP_ID" --matrix --last 5 \
    > "$matrix_tmp" 2>&1 || echo "(matrix unavailable)" > "$matrix_tmp"
else
  echo "(python or root missing)" > "$matrix_tmp"
fi
mv "$matrix_tmp" "$OUT_MATRIX"

dash_tmp=${OUT_DASH}.tmp.$$
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
  echo "| phase | \`$phase\` |"
  echo "| next_action | \`$next_action\` |"
  echo "| blocker_count | \`$blocker_count\` |"
  echo "| heartbeat | \`$heartbeat\` |"
  echo
  echo "Status file: \`$OUT_STATUS\`  "
  echo "Matrix file: \`$OUT_MATRIX\`  "
  echo "Log: \`/tmp/autotrain-continuous-driver.log\`"
  echo
  echo "## Skill matrix (last 5 cycles)"
  echo
  matrix_bytes=$(wc -c < "$OUT_MATRIX")
  if [ "$matrix_bytes" -le 65536 ]; then
    cat "$OUT_MATRIX"
  else
    echo "(matrix exceeded 64 KiB compact-report ceiling: ${matrix_bytes} bytes)"
  fi
} > "$dash_tmp"
mv "$dash_tmp" "$OUT_DASH"

# also print to stdout for agents to paste
cat "$OUT_DASH"
