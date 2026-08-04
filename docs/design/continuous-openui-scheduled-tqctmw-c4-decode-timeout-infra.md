# Autotrain c4 (continuous-openui-scheduled-tqctmw): decode timeout, measurement incomplete (infra, non-positive)

**Verdict:** infrastructure-incomplete measurement, not scoreable. Both the
`control` and `component-edge` `wf_smoke_v2` arms (1,766,987 params, seed
100004) finished training, but every one of the 3 smoke records timed out
during decode (`smoke:decode_timeout_count actual=3 need=0`,
`incomplete_document_n=3`) — no arm reached a scoreable
`structural_similarity`/`parse_rate` (`primary_metric_unavailable`). This is
not a model result.

Machine evidence:
[`continuous-openui-scheduled-tqctmw-c4-decode-timeout-infra.json`](continuous-openui-scheduled-tqctmw-c4-decode-timeout-infra.json).

## Observation (not yet attributed)

`suites.smoke.compiler_ms_mean` jumped from **~10,000 ms** in c3 (integration
commit `14190af4`, pre-merge) to **~23,000 ms** in c4 (integration commit
`8e80aa5e`, after merging `origin/main` `5ba8e430` — the
[precompiled-grammar-admissibility](precompiled-grammar-admissibility-20260804.md)
campaign's L1 landing touching `compiler_draft.py` and
`completion_kernel.py`) — roughly 2.2×, comfortably exceeding the 8 s
screening decode timeout
(`climb_policy.decode_timeout_seconds_for_role`, `screening` default).

This coincides with, but is **not established as caused by**, that merge:
this sandbox also ran a fresh `npm ci` and two prior train+eval cycles
earlier in the same session, which is a confound for wall-clock comparison
on a shared, noisy CPU box. Recorded as an observation for the next session
to attribute with a controlled before/after profile
(`scripts/profile_generate` / `bench_telemetry`), not as a claimed
regression.

## Delivery

Non-positive (`measurement_incomplete`, `harness_failure`,
`primary_metric_unavailable`): no stack layer. Handoff requests
`repair_harness` (`model_build`) before the next `retry_measurement` of this
frozen arm (`frozen_manifest_sha256`
`0b30e0a64dc96fdc3824c105eadb7f64cd72e807866da78d4e85567b24bfe039`).
Deferred to the next continuous-loop session: attribute the `compiler_ms`
jump with a controlled profile before choosing between (a) recalibrating the
screening decode timeout (precedent: PR #1403, 8s→10s) or (b) filing a
`HarnessSignalV1` against the precompiled-grammar-admissibility campaign if
profiling confirms a real regression.
