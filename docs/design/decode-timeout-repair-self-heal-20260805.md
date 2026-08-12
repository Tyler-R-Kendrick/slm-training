# Decode-timeout `repair_harness` self-heal (continuous)

**Honesty:** harness continuous-driver design. Not a ship claim.

## Problem

Soft-clear of `state.json` BLOCKED did not advance continuous thrash when the
predecessor handoff still had an unacknowledged `repair_harness` for AgentV /
compiler_ms decode timeouts. Forever cleared the flag; the driver re-raised
`unacknowledged actions: 0:repair_harness`; after three failures the loop
BLOCKED again. `_self_heal_cycle_error` only handled bank exhaust / feedback
identity / knob signature races.

## Design

When `_self_heal_cycle_error` sees that error and the pending `repair_harness`
reason matches decode-timeout markers:

1. Ensure a **post-integration** git commit exists (stamp design receipt if tip
   already has dual-arm / residual routing code).
2. Append a completed `ack-action` receipt with that commit as evidence.
3. Rewrite any `retry_measurement` freeze-replay action to `next_experiment`
   and prefer decode residual arms (`bounds` / `canvas` / `both` /
   `cached-compiler-decision-margin`).

Formal / Lean `repair_harness` reasons are **not** auto-acked.

## Scope

- In: continuous driver self-heal for decode-timeout repair prereqs.
- Out: inventing new decode kernel speedups; unconstrained decode; auto-ack of
  formal/data/delivery prerequisites.

## Tests

- `test_parse_unacked_predecessor_campaign`
- `test_is_decode_timeout_repair_reason`
- `test_self_heal_cycle_error_decode_timeout_repair_harness`
- `test_rewrite_decode_timeout_handoff_to_next_experiment`
