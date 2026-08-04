# Autotrain continuous-openui-local c2 (2026-08-04): component-plan decode-timeout harness stop

**Verdict:** measurement incomplete, no model attribution. Harness repair
landed; frozen arm replay required before any new hypothesis.

Cycle 2 of `continuous-openui-local`, started clean from integration commit
`e55955b` (this loop's own cycle-1 documentation commit). Ran the ranked
successor from cycle 1 — matched `control` vs `component-plan` arms, both
attaching `structural_aux_head_profile="component-plan"` for parameter
parity (1,755,764 params each, up from cycle 1's 1,608,960) with the
candidate additionally setting `component_plan_decode_weight=1.0`.

Both arms **completed training** but AgentV reported
`smoke:decode_timeout_count=3/3` — every one of the 3 smoke documents hit an
internal decode timeout, so `meaningful_program_rate`, `structural_similarity`,
and every other quality metric came back `null`. This is a measurement
failure, not a quality result: `expected_gate_rejection=false`,
`gate_failures.measurement_integrity_failures` covers all seven quality
criteria, and `gate_failures.runtime_failures` reports
`smoke:decode_timeout_count actual=3 need=0`.

Root cause (see
[`autotrain-continuous-openui-local-decode-timeout-repair-20260804.json`](autotrain-continuous-openui-local-decode-timeout-repair-20260804.json)):
`compiler_ms_mean` was ~23,151ms (control) / ~23,199ms (candidate) — nearly
identical between arms even though `component_plan_decode_weight` is `0.0`
in control, so the cost tracks with the aux head being **attached**
(`structural_aux_head_profile != "none"`), not with the decode-weight lever
that scores it. The prior `screening_decode_timeout_seconds=8` policy default
produced a `8s x smoke_n(3) = 24s` chunk budget — under a second of margin
over the ~23.2s observed cost, so ordinary CPU jitter in this scratch/CPU
container timed out every record.

Repair: raised `screening_decode_timeout_seconds` 8s → 10s in
[`policy.v1.json`](../../src/slm_training/resources/experiments/autotrain_climb/policy.v1.json)
(`harness.autoresearch.experiment_campaign` v177 → v178), still subject to
`_fit_screening_decode_timeout_seconds`'s arm-wall clamp so cheap screening
arms are unaffected. Added
`test_screening_decode_timeout_covers_observed_aux_head_cost` in
`tests/test_scripts/test_run_autotrain_continuous.py` asserting the fitted
chunk budget exceeds the observed 23.2s cost with a >1s margin. Repair
commit: `a17419c` (`fix(autotrain): raise screening decode timeout 8s->10s
for aux-head arms`).

Per continuous-loop law, the identical frozen arm
(`continuous-loop-20260804-continuous-openui-local-8c0b60dd-c2`,
`frozen_manifest_sha256=2fd7771dab610c8ddc6c6d32cba0eedfc24bd549471b7d946b26d7be5df70581`)
must be replayed under the repaired timeout before any new hypothesis is
queued. No checkpoint from this incomplete cycle is promotable, reusable, or
ship-eligible.

Machine evidence:
[`autotrain-continuous-openui-local-decode-timeout-repair-20260804.json`](autotrain-continuous-openui-local-decode-timeout-repair-20260804.json).
