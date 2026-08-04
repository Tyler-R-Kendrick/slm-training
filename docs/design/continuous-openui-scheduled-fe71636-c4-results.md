# Continuous autotrain: 2026-08-04 (scheduled loop `fe71636`) cycle 4 — decode-timeout budget increase insufficient

**Loop:** `continuous-openui-scheduled-fe71636`
**Campaign:** `continuous-loop-20260804-continuous-openui-schedu-3d42338c-c4`
**Integration commit:** `c5d5bee7` (cycle 3's decode-timeout repair + docs commits merged)
**Replay of:** [`continuous-loop-20260804-continuous-openui-schedu-3d42338c-c3`](continuous-openui-scheduled-fe71636-c3-results.md)

**Verdict:** the `screening_decode_timeout_seconds` `8 → 12` repair (commit
`3c30288`) did **not** change the outcome.

| Cycle | control compiler_ms_mean | canvas compiler_ms_mean | decode_timeout_count |
| --- | ---: | ---: | --- |
| c3 (8s config) | 23080.2 | 23127.6 | 3/3 both arms |
| c4 (12s config) | 23042.1 | 23173.9 | 3/3 both arms |

The measurements are statistically indistinguishable between the two
configs. This rules out "8s was just barely too tight" as the explanation:
the real per-record compile+decode cost on this CPU-only sandbox (~23s) is
roughly **2x** even the raised 12s-per-record (36s combined) budget, not a
narrow miss the earlier bump could close.

## Honest assessment

This is now **two consecutive cycles** (c3, c4) with the identical
decode-timeout blocker and consistent measurements. Per the `autotrain`
continuous-mode loop law, a third identical recurrence with no new
information would be the repeated-hard-block threshold. This session stops
here rather than force a third replay attempt or guess another timeout
value without evidence it would help — timeout-value tuning is exhausted
given the ~2x gap.

## Next levers for a future session (not attempted this session)

1. Reduce `screening_smoke_n` below 3 so fewer records must fit the
   per-cycle eval wall.
2. Reduce screening training steps (currently 20) to leave more of the
   shared `MAX_RUN_MINUTES=3` budget for eval.
3. Accept that `wf_smoke_v2` screening does not fit `MAX_RUN_MINUTES=3` on a
   CPU-only sandbox of this class, and treat this loop's screening role as
   diagnostic-only (harness/infra work, not model climb) until faster
   compute is authorized.

## SDLC Phase A

**Non-positive** (`measurement_incomplete`, `harness_failure`,
`fixture_insufficient_n` on both arms). No stack layer. The `repair_harness`
action for this cycle is **left unacknowledged** — the next continuous-loop
invocation should pick one of the three levers above rather than repeat the
same timeout-bump approach, matching this loop's own diagnosed conclusion
that timeout tuning alone is exhausted.

## This session's delivery

Across cycles c2-c4 this session landed two genuine, tested, evidence-backed
harness repairs even though neither cycle reached a "positive" model result:

- `2aedf3b` — AgentV SDK self-heal bootstrap (`npm ci` on missing SDK),
  proven executable-unblock: c2's `RuntimeError` crash became c3's complete
  `gates.json` verdict.
- `3c30288` — `screening_decode_timeout_seconds` `8 → 12`, evidence-based
  but honestly documented here as insufficient on its own.

Per `sdlc` autotrain-iteration-delivery, neither is a "positive" (primary
metric win / ship-quality win / full executable unblock to a *usable*
scoreboard) result, so no stacked layer is claimed for a training win — this
session's code lands as a plain harness-fix PR, matching the established
pattern of prior sessions' infrastructure PRs on this loop (`#1403`,
`#1406`, `#1410`, `#1420`, `#1423`).
