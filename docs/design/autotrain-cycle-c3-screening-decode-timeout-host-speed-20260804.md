# Autotrain c3 (continuous-openui-local, 2026-08-04 session): screening decode timeout is a host-speed limit, not a code defect

**Verdict:** no code or policy change. Cycle
`continuous-loop-20260804-continuous-openui-local-8c0b60dd-c3`
(frozen `manifest_sha256=74be089bb2215fd0db676016cd0b8aec2d5e21a9fa48fcd2a595dd25061555d9`)
trained both `control` and `canvas` `wf_smoke_v2` arms (1,608,962 params, 22
steps) fine, and the AgentV eval path now runs end to end (the cycle-2
AgentV-SDK-missing gap is fixed). But the smoke suite only produced
`n=3` documents and all 3 hit the per-record decode timeout:
`suites.smoke.compiler_ms_mean≈23127–23448ms` against
`--decode-timeout-seconds 8.0`, so `smoke:decode_timeout_count actual=3
need=0` and `smoke:insufficient_n actual=3 need>=20` — the same shape as
`docs/design/autotrain-cycle-c3-bounds-quality-neutral.md` /
`continuous-openui-20260802-c3-results.md`.

## Why this is not a bug

`decode_timeout_seconds_for_role` (`src/slm_training/autoresearch/climb_policy.py:347`)
returns the `screening_decode_timeout_seconds` policy default of `8.0`, and
`_fit_screening_decode_timeout_seconds`
(`scripts/run_autotrain_continuous.py:261`) only ever clamps that value
*downward* to fit the `MAX_RUN_MINUTES=3` arm-wall budget — it does not
measure or adapt to actual host compile speed. `thrash_timing.json` for this
cycle confirms `clamp_bound=0.0`, i.e. the fit did **not** shrink the
configured `8.0s`; the eval simply ran into a wall the config assumed would
not be hit.

`8.0s` was deliberately locked in
[`autotrain-thrash-timing-pareto-20260803.md`](autotrain-thrash-timing-pareto-20260803.md)
from real incomplete-rate telemetry (`3×8=24s` eval budget under a ~70s arm
share), with an explicit **non-goal**: *"Raising `MAX_RUN_MINUTES` as default
thrash fix"* and a locked rule: *"Never ad-hoc wall++ because a cycle
failed. Never silent widen mid-campaign."* This ephemeral remote-execution
container (4 vCPU Intel Xeon @ 2.80GHz, freshly bootstrapped `.venv` +
`node_modules` this session) is simply producing a per-document compiler wall
time (~23s) roughly 3x what the locked calibration assumed for a screening
probe. One cycle's timeout on one (possibly colder-than-steady-state, since
this was the container's first `evaluate_model` invocation after a from-fresh
`npm ci`) host is exactly the "single cycle failed" case the locked policy
says **not** to react to — real recalibration needs accumulated
`thrash_timing.jsonl` incomplete-rate telemetry, not one data point.

## Disposition

- No code change, no policy/knob change — `screening_decode_timeout_seconds`
  stays `8.0`, consistent with the locked Pareto calibration.
- This is an environment/host-speed characteristic of this specific
  container, not a `model_build` harness defect. Filed as
  `repair_harness` evidence per the continuous-driver handoff contract
  (`frozen_manifest_sha256=74be089bb2215fd0db676016cd0b8aec2d5e21a9fa48fcd2a595dd25061555d9`)
  so the queued `retry_measurement` can proceed; the replay is expected to
  reproduce the same `insufficient_n` outcome deterministically (identical
  steps/decode-timeout/seed), which is itself informative telemetry, not a
  blocker — soft failures (timeouts, fixture `n`) never stop the continuous
  loop (`.claude/skills/autotrain/references/continuous.md` "Absolute loop
  law", rule 4).
- If `thrash_timing.jsonl` telemetry across multiple sessions on
  similarly-provisioned containers shows a persistently high incomplete rate
  (the Pareto table's "High (≫15%)" band), that is the trigger for a real,
  evidence-bound recalibration of `screening_decode_timeout_seconds` /
  `screening_thrash_steps` — not this single finding.

Lean is `not_applicable:screening`; climb `inconclusive`; ship `blocked`.
No checkpoint promotion or ship claim is made from this cycle.
