# E686 — per-row decoder trace budget

Date: 2026-07-21
Status: completed negative; rejected; not ship

E686 tested whether a per-row constrained-selection trace budget could stop
early Held-out records from consuming the global telemetry allowance. The
independently capped full Held-out replay completed with exit 0, no timeout or
fallback, and emitted AgentEvals JSONL plus an AgentV SDK bundle.

Quality is prediction- and metric-identical to E685: strict v2.6.0 remains
2/5, structure 0.5108, component recall 0.5733, reward 0.8602, and AgentV 0/1.
The observability hypothesis also fails. The aggregate contains 98 selection
traces, all labeled `row=0`, and no identifiable tabs trace.

The cause is architectural: this evaluator generates one record per model
call, so the model-local row is always zero. `eval_runner` later concatenates
the independent `DecodeStats` objects without attaching the source record ID.
Increasing or partitioning the model-local budget cannot recover that identity
and only inflates the first record's telemetry.

Reject and revert v140. The next lever must preserve the original per-call
bound and annotate each trace with its record ID at the eval aggregation
boundary. No checkpoint was created, synced, or promoted.

Evidence: [JSON](iter-e686-per-row-trace-budget-20260721.json).
