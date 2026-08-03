# Autotrain c3 (continuous-openui-local): control decode timeout, measurement incomplete

**Verdict:** not scoreable, not a model result — soft failure. Both arms
trained (`control` 20 steps, loss `15.2976`, checkpoint `07b25357...62043e0`;
`component-plan` 20 steps, loss `19.4065`, checkpoint `8e25e2a0...810752c0`),
but the `control` arm's smoke evaluation had all 3 documents hit the 8s
screening decode timeout (`smoke:incomplete_document_n=3:decode_timeout_count=3`),
so no `structural_similarity` (or any smoke metric) exists for `control`.
`component-plan` completed evaluation but scored `meaningful_program_rate=0.0`
on its 3 documents — `executable_unblock_rejected_low_mpr`.

Because the primary metric is unavailable for `control`, the arms are not
comparable this cycle (`primary_metric_unavailable`). This is a single
timeout — a soft failure under the continuous-loop contract, not a repeat or
hard blocker, and not attributed to the `missing_dev_env_prerequisites()`
preflight guard added in c1 (the AgentV SDK ran cleanly here; this is a
decode-latency timeout, a different failure mode).

Rejected as `fixture_insufficient_n_alone` combined with the incomplete
control measurement; no stack layer for this cycle. Lean is
`not_applicable:screening`; checkpoints are local, explicit no-sync, never
reusable/promoted/synced/ship.

Next: `retry_measurement` — replay the exact frozen control/component-plan
pair once to see whether the timeout reproduces (recipe/wall issue) or was
noise (retry lands inside the budget).

Machine evidence:
[`autotrain-cycle-c3-component-plan-control-decode-timeout-20260803.json`](autotrain-cycle-c3-component-plan-control-decode-timeout-20260803.json).
