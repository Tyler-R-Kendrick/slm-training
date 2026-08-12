# Dual-arm decode timeout escape (continuous thrash)

**Honesty:** harness repair for continuous fixture thrash. Not a ship claim.

## Problem

When both control and candidate arms hit `decode_timeout` under
`compiler_ms`-dominant cost, the handoff emitted `repair_harness` +
`retry_measurement` with a frozen dual-arm seed. That freeze-replay loop
never advances thrash and left the forever supervisor stuck on unacked
`repair_harness` after c18–c20 dual-arm timeouts.

## Fix

1. Dual-arm decode timeouts emit `repair_harness` + **`next_experiment`**
   (decode residual trajectory), not frozen dual-arm `retry_measurement`.
2. Predecessor priority prefers decode residual slugs
   (`bounds` / `canvas` / `both` / `cached-compiler-decision-margin`) when the
   prior cycle was dual-arm / compiler_ms incomplete.
3. Candidate-only timeouts still use the existing freeze-replay budget path.

## Evidence

- Worktree continuous-openui-local c18–c20 dual-arm timeout handoffs.
- `timeout_dominant_phase=compiler_ms` on smoke_hero_01.
