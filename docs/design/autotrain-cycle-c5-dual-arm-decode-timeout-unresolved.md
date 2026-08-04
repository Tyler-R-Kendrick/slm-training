# Autotrain c5: dual-arm decode timeout, root cause unresolved (finding, not a fix)

**Verdict:** infrastructure failure, not scoreable, repair still required. The
c4 champion's fresh-seed confirmation (seed `100005`, same 1,755,760-param
`component-plan` recipe) finalized all 3/3 smoke records inside a typed
decode timeout for **both** the control and confirm arms
(`decode_timeout_document_count=3`, `effective_decode_timeout_seconds_max`
`24.0` / `23.43`). The same size candidate decoded successfully in ~6.5-7.2s
per arm one cycle earlier (c4, different seed) — this is not a chronic
this-size-never-works problem, but it is not explained by this session's
investigation either.

What was checked and ruled out:

- `_effective_record_decode_timeout`
  (`src/slm_training/harnesses/model_build/eval_runner.py:884`) allocates a
  fair, budget-aware per-record timeout share and is already covered by
  `tests/test_harnesses/model_build/test_eval_metric_semantics.py::test_eval_wall_fairly_caps_each_remaining_record`.
  No defect found in the allocator itself.
- A routing change was drafted to auto-retire a *symmetric* (both-arm)
  finalized decode timeout after the frozen-replay limit, mirroring the
  existing asymmetric `candidate_only`/`control_only` reproduce-then-retire
  paths. It was **reverted, not shipped**: it directly contradicted the
  existing, deliberately-tested contract in
  `tests/test_scripts/test_run_autotrain_continuous.py::test_replayed_dual_arm_timeouts_remain_inconclusive_and_require_repair`,
  which locks in that a dual-arm timeout must stay `inconclusive` and keep
  demanding `repair_harness` rather than silently retiring to a new
  hypothesis — because when *both* matched arms fail identically, that is
  evidence the harness itself may be unreliable, and generating more results
  under an unreliable harness would be meaningless. That guardrail is correct
  and this session did not have enough evidence to justify weakening it.

Root cause remains **unresolved**: either (a) seed `100005` triggers
seed-dependent worst-case decode/parser behavior at this candidate size, or
(b) this sandbox's CPU throughput combined with the per-experiment wall
budget leaves too little decode headroom at 1,755,760 params. This session
did not distinguish between the two.

Checkpoints (`769e7d46...db78a66` control, `6d9ff92f...b073b9fe4` confirm) are
local, explicit no-sync, and not reusable, promotable, or ship. The champion
(`champ-continuous-openui-local-4-2694d77fc99953e4`) is not confirmed and not
promoted pending resolution.

Next: replay the identical frozen `-confirm` arm (`retry_measurement`,
already queued in the handoff) to test whether the timeout reproduces at the
same seed. If it reproduces twice, hand off to `improve-openui-harnesses` for
dedicated compiler-tree decode profiling at seed `100005` and this param
count — that is real investigative work this session's scope and time
budget did not stretch to.

Machine evidence:
[`autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.json`](autotrain-cycle-c5-dual-arm-decode-timeout-unresolved.json).
