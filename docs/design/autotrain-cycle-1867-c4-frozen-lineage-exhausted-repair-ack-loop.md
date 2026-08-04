# Autotrain c1867: c2 frozen lineage exhausted; repair-ack does not force a fresh hypothesis

**Verdict:** genuine repeated-blocker — three consecutive cycles (c2, c3, c4)
on the identical `component-plan`/`control` decode-timeout, with the third
occurrence surfacing a driver-level finding distinct from the original
recalibration.

Campaign `continuous-loop-20260804-continuous-openui-schedu-6699f447-c4`,
cycle 4, `intent=retry_measurement` again. `effective_decode_timeout_seconds`
is still `24.0` even though the integration commit (`7dedb8e5`) is well after
the c1865 recalibration (`eecb8304`, `screening_decode_timeout_seconds`
`8→10`) — because this is, for the third time, a replay of the **same**
`continuous-loop-...-c2` frozen manifest, which pins its own
`decode_timeout_seconds=8` at freeze time and never picks up a later policy
change (documented as expected in c1866).

**New finding this cycle:** acknowledging the `repair_harness` action resets
`max_consecutive_frozen_replays` (currently `1`) and lets the driver queue
*another* replay of the identical frozen lineage — but the frozen manifest
itself never changes, so every replay reproduces the identical
`decode_timeout_count=3/3` failure and the driver re-requests another
`repair_harness` action. Acking repeatedly (c1865 → c1866 → this cycle) does
not converge: there is no code left to fix (the c1865 fix is correct and
already verified), yet the hypothesizer's top-ranked priority keeps choosing
"replay the exact frozen control and candidate before testing a new
hypothesis" instead of retiring this lineage and composing a genuinely fresh
`component-plan` experiment id that would pick up the current policy.

This session stops feeding acknowledgments into this specific loop here,
per the autotrain continuous-mode law: the same hard blocker (decode timeout
on this frozen `component-plan` lineage) has now failed three consecutive
cycles (c2, c3, c4) and further repair acks do not recover it — recovery
requires a harness change to the hillclimb/hypothesizer's frozen-replay
exhaustion path (e.g., retiring a lineage after N replay attempts and forcing
a fresh, non-`replay_of_manifest_sha256` hypothesis), which is out of scope
for this cycle's repair and is flagged here as the next harness-family
`experiments`/`autoresearch` finding for a follow-up session.

Classified `SDLC_PHASE_A NON_POSITIVE`. No stack layer; local commit only.
`repair_harness` for this cycle is **not** acknowledged (nothing new to
repair); the loop's next action is a fresh, non-replay hypothesis or a
harness change to the exhaustion path, whichever a follow-up session reaches
first.

Machine evidence: [`autotrain-cycle-1867-c4-frozen-lineage-exhausted-repair-ack-loop.json`](autotrain-cycle-1867-c4-frozen-lineage-exhausted-repair-ack-loop.json).
