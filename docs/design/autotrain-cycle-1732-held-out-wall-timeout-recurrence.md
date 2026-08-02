# Autotrain c1732: held_out wall-timeout recurs on frozen replay (2nd occurrence, still soft)

**Verdict:** the driver's automatic frozen-replay of c1731's timed-out
promotion campaign (`c5`, same frozen manifest as `c4`) hit the identical
`stage exceeded wall-time limit` failure for
`scripts.evaluate_model --suites smoke,held_out` on both the `control` and
`steps` arms. This is occurrence **2 of 3** toward the loop's hard-block
threshold — still a soft failure per the continuous-mode loop law, and the
campaign never produced a `cycle_handoff.json` (the driver's own trailing
`status --matrix` call was killed too, at 7.4s, because the two arm timeouts
had already spent the cycle's wall budget). This is an initialized-only
campaign; the driver's recovery logic will skip past it on the next
supervised invocation, the same way `c2` was skipped in favor of `c3` earlier
in this loop.

## Root cause (unchanged from c1731)

CPU-only constrained decode for the combined `smoke,held_out` suite set does
not fit inside the repository-wide 3-minute `MAX_RUN_MINUTES` cap on this
container. `held_out` records are longer than the 3-record `smoke` fixture.
Screening-role cycles (c1729/c1730), which request `smoke` only, are
unaffected and complete well inside the cap.

## Next-run priority

If this exact `stage exceeded wall-time limit` on `smoke,held_out` recurs a
third consecutive time with no new information, that crosses the hard-block
threshold for promotion-role cycles specifically. The correct repair is **not**
to drop `held_out` from promotion suites or weaken ship gates — both are
forbidden. The correct repair, routed through `improve-openui-harnesses`, is
to split the promotion-role `evaluate_model` invocation into two
independently wall-budgeted stages (`smoke` then `held_out`) instead of one
combined `--suites smoke,held_out` process, so each stage fits inside
`MAX_RUN_MINUTES` on its own rather than the combined call exhausting it in
one shot.

No checkpoint, model-card, or README change applies to this cycle.
Machine-readable evidence is in
[`autotrain-cycle-1732-held-out-wall-timeout-recurrence.json`](autotrain-cycle-1732-held-out-wall-timeout-recurrence.json).
