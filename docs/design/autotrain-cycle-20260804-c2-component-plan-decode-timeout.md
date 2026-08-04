# Autotrain cycle 2 — component-plan decode timeout (2026-08-04)

Scheduled continuous-loop cycle 2 (`continuous-openui-local`,
`continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`) tested the
size-matched component-plan hypothesis (1,755,764 params, control vs.
`component_plan_decode_weight` candidate) recommended by cycle 1's priorities.

**Measurement did not complete.** All 3 smoke documents decode-timed-out on
*both* arms (`completed_document_n=0`, `decode_timeout_count=3`), so there is
no attributable model signal from this cycle — it is an infrastructure
result, not a rejected or positive model result.

| Arm | document n | completed | decode timeouts | compiler ms (mean) |
| --- | ---: | ---: | ---: | ---: |
| Control | 3 | 0 | 3 | 23342.9 |
| Component-plan candidate | 3 | 0 | 3 | 23275.8 |

## Diagnosis

The continuous driver's handoff flagged a `repair_harness` action
(`harness_family=model_build`) because AgentV finalized every record
disposition and typed the failure as a decode timeout. Investigating before
touching any code:

- Screening's fitted per-record decode budget is `8.0s`
  (`_fit_screening_decode_timeout_seconds`, confirmed from
  `matrix-proposal.json`'s `decode_timeout_seconds=8.0` for every hypothesis
  in this campaign) — well under the `<=12.0s` thrash-calibrated ceiling
  asserted by `test_fit_screening_decode_fits_arm_wall`.
- The scoreboard's `effective_decode_timeout_seconds_min/max=24.0` is the
  **whole-chunk** budget (`8.0s x decode_batch_size_max=3 = 24.0`), not a
  per-record timeout — this was initially misread during triage as "the
  per-record budget was loosened to ship's 24s," which is not what happened.
- The component-plan recipe's `compiler_ms_mean` (~23.3s) sits right at that
  24s chunk budget on this container's CPU throughput. This matches the
  historical 2026-08-03 run of the *same* recipe family
  ([results](autotrain-cycle-20260803-c2-component-plan-screen.md)), which
  reported p50 latencies of 18.3–23.1s with **no** code change in between —
  i.e. this recipe has always run close to this edge; today's container
  throughput pushed it over.

**Conclusion: not a harness code defect.** No timeout, gate, or budget was
loosened. The repair delivered is a clarity fix: a comment at the
`effective_decode_timeout_seconds_min/max` computation in
`eval_runner.py` plus a pinned regression test
(`test_screening_chunk_timeout_is_per_chunk_not_per_record`,
`tests/test_harnesses/model_build/test_eval_metric_semantics.py`) that locks
in the exact `decode_timeout_seconds=8.0`, `chunk_record_n=3` → `24.0`
scenario, so a future diagnosis doesn't repeat this same misread.

## Next

Per the handoff's `retry_measurement` action, replay the identical frozen
arm (`frozen_manifest_sha256=2ea993ec6cd66682c2bf5e47214af8f85f4d720e55e0e0fc0223e77e5752b43a`)
next cycle. If timeouts recur across further replays, the honest escalation
is a per-family chunk-size reduction (so one slow record doesn't sink its
chunk-mates) — never a blanket `decode_timeout_seconds` increase.

Both checkpoints
(`outputs/autoresearch/continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2/runs/`)
are local-only, unevaluated fixture artifacts — never promoted/synced/ship-eligible.

JSON twin: [autotrain-cycle-20260804-c2-component-plan-decode-timeout.json](autotrain-cycle-20260804-c2-component-plan-decode-timeout.json)
