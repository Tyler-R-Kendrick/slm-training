# Autotrain c1866: c3 frozen replay does not pick up the c1865 recalibration (expected)

**Verdict:** not a new blocker. Frozen replays intentionally reproduce the
identical arm under its originally pinned config, not the current policy.

Scheduled continuous loop `continuous-openui-scheduled-72bc0d`, campaign
`continuous-loop-20260804-continuous-openui-schedu-6699f447-c3`, cycle 3. This
was the driver's `retry_measurement` of c2's frozen `component-plan`/`control`
arms after the c1865 repair (`screening_decode_timeout_seconds` `8→10`,
commit `eecb8304`).

Both arms again hit `decode_timeout_count=3/3`, and
`effective_decode_timeout_seconds_min/max` on the c3 scoreboards is still
`24.0` (`= 8s × 3`, the **pre-repair** chunk budget), not the recalibrated
`30.0` (`= 10s × 3`). This is by design: a frozen replay reproduces the
identical arm/manifest, pinned config included, so a policy change made after
the manifest was frozen does not retroactively apply. This is the same
frozen-replay semantics this repository documented previously in open PR
#1403 (`c6` there hit the identical situation).

Because `measurement.max_consecutive_frozen_replays=1` was already spent on
this c2→c3 replay, the driver requested another `repair_harness` action. No
new code change is needed or made — the c1865 fix is already correct and
verified by its regression test; it will apply on the next **fresh**
(non-replay) hypothesis. This cycle's `repair_harness` action is acknowledged
against the same c1865 commit (`eecb8304`) since there is nothing further to
repair; the actual validation moves to the successor cycle, which the driver
should route to a new `component-plan` hypothesis (not a replay) now that the
frozen-replay bank is reset.

Classified `SDLC_PHASE_A NON_POSITIVE` (`measurement_incomplete`,
`harness_failure`, `fixture_insufficient_n_alone`, `primary_metric_unavailable`).
No stack layer; local commit only.

Machine evidence: [`autotrain-cycle-1866-c3-frozen-replay-does-not-pick-up-recalibration.json`](autotrain-cycle-1866-c3-frozen-replay-does-not-pick-up-recalibration.json).
