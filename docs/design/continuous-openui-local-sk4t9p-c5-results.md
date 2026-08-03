# Continuous autotrain: 2026-08-03 (scheduled session sk4t9p) cycle 5 — component-plan fresh-seed confirmation inconclusive (known decode-timeout blocker)

**Loop:** `continuous-openui-local`
**Campaign:** `continuous-loop-20260803-continuous-openui-local-8c0b60dd-c5`
**Integration commit:** `23fc0b3d` (this session's harness-repair commit)

**Verdict:** the driver's queued fresh-seed confirmation of the
`component-plan` champion (rank-1 priority from cycle 2) trained both arms
to completion, but the `confirm` arm's evaluation hit **3 decode timeouts**
and never produced smoke quality metrics — the exact same seed-`100005`
dual-arm decode timeout with an undetermined root cause first documented in
[`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)
and explicitly deferred by every prior `continuous-openui-local` session.

| Arm | Seed | Status | structural_similarity | binder_reference_f1 | p50 ms |
| --- | ---: | --- | ---: | ---: | ---: |
| control | 100005 | complete | .04307 | .82222 | 6947.6 |
| confirm | 100005 | decode_timeout_count=3 | — | — | — |

Both checkpoints exist on disk (control
`95073bc3...1a3712d`, confirm `38392edb...49f4b9c59`) — training completed
for both arms. Only the confirm arm's AgentV evaluation stage failed.

## This reproduces a known, deliberately-deferred blocker

This is **not** a new bug. It is the same seed-`100005` decode timeout that:

- Blocked the `component-plan` champion's fresh-seed confirmation in an
  earlier session ([`autotrain-cycle-c5-c6-replay-blocked-follow-up.md`](autotrain-cycle-c5-c6-replay-blocked-follow-up.md)),
  which explicitly asked for a **dedicated `improve-openui-harnesses`
  session**.
- Was distinct from the `-confirm`-suffixed arm-slug recognition bug that
  PR #1370 (`fix(autotrain): complete fresh-seed confirmation flow`) already
  fixed — that fix let this cycle run at all (frozen-replay slug matching
  now works), but the underlying decode timeout itself persists.
- Is **not** related to this session's earlier harness fix (the
  `_arm_execution_deadline` hypothesize-feedback race, `harness_family=autoresearch`)
  — the driver's `cycle_handoff.json` for this cycle names a different owning
  family, `harness_family=model_build`, with `frozen_manifest_sha256`
  `6ee028aa...cc83e7b`.

Per the autotrain non-negotiable rule "never mix harness and model changes
in one attribution arm," this session does not attempt an ad-hoc fix for the
`model_build` decode timeout — consistent with every prior session's choice
to defer it to dedicated investigation time.

## SDLC Phase A

**Non-positive** (`measurement_incomplete` + `harness_failure`). Champion
queue status: `confirmation_inconclusive`. No stack layer for this cycle.

## Next priorities

1. Do not speculatively re-attempt the seed-`100005` dual-arm decode
   timeout; route to a dedicated `improve-openui-harnesses` session against
   `harness_family=model_build` (frozen manifest `6ee028aa...cc83e7b`).
2. Once repaired: replay the identical frozen `c5-confirm` arm before any
   new model hypothesis.

Machine evidence:
[`continuous-openui-local-sk4t9p-c5-results.json`](continuous-openui-local-sk4t9p-c5-results.json).
