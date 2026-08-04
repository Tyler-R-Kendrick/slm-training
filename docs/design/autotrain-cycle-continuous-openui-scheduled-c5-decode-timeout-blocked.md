# Autotrain c5 (continuous-openui-scheduled): decode-timeout blocked (3x repeat)

**Verdict:** formally `blocked` on this sandbox for the frozen c2 arm's smoke
measurement — three consecutive cycles (c3, c4, c5) reproduced the identical
decode-timeout failure with no new information, meeting the loop law's
repeated-hard-blocker threshold
(`.claude/skills/autotrain/references/continuous.md` §4).

| Cycle | control compiler_ms_mean | canvas compiler_ms_mean |
| --- | --- | --- |
| c3 | 23,165.5 | 23,242.0 |
| c4 | 23,216.8 | 23,100.9 |
| c5 | 21,295.7 | 23,279.4 |

All three runs: `decode_timeout_count=3/3`, fitted
`screening_decode_timeout_seconds≈8.0`, on the identical frozen manifest
(`00574c47f6362eaae01b999e28683e721f092026d1c561e5669ef22a0a811210`). The
spread across cycles (21.3s-23.3s) is normal run-to-run variance around a
consistent ~22s mean, not the wide swing a transient contention spike would
produce (`/proc/loadavg` stayed uncontended throughout, per c4's doc).

## Disposition (loop law §4)

> Stop only when blocked. Report blocked only after the same hard blocker has
> failed three consecutive cycles with no new information **and**
> in-pipeline self-heal could not recover.

Both conditions are met:

- **Three consecutive, no new information**: c3 → c4 → c5 all fail the same
  way at the same magnitude. c4 already ruled out cold-start; c5 adds no new
  hypothesis, only confirms the range.
- **Self-heal could not recover**: the AgentV-SDK-missing blocker (c2) *was*
  self-healed (commit `1c48ac9`) and the design_md/torch-leak defects found
  while investigating *were* fixed (commit `3ce1b58`) — those are exactly the
  in-pipeline recoveries the loop law asks for. What remains is not a code
  defect: this sandbox's decode throughput is structurally too slow for the
  `screening` role's wall-budget arithmetic
  (`3 records × ~22s + 20s train floor + 8s overhead ≈ 86s` against an
  `arm_wall_seconds≈52.76s` share of the 180s `MAX_RUN_MINUTES` cap). No
  timeout-knob value fixes that; it needs either faster compute or a
  deliberately reviewed thrash-recipe change backed by multi-sandbox
  evidence (out of scope for a single scheduled-routine session per
  `_fit_screening_decode_timeout_seconds`'s own docstring warning against
  "silent wall++").

## Closeout actions taken

- Stopped starting new cycles against this frozen arm on this sandbox.
- `sdlc` Phase B check: no autotrain stack layers were ever opened this
  iteration (every cycle classified `NON_POSITIVE`, correctly per
  `autotrain-iteration-delivery`'s positive-result gate), so there is nothing
  to inventory or merge — the delivery for this iteration is the four local
  commits plus PR #1410 (infra fixes + honest diagnostics, not a training
  result).
- This loop (`continuous-openui-scheduled`) remains resumable: a future
  scheduled firing on this sandbox class should skip straight to a genuinely
  new hypothesis/campaign rather than replaying this frozen arm again, or run
  on faster compute if the goal is specifically to complete this measurement.

Machine evidence: cycle campaigns under
`outputs/autoresearch/continuous-loop-20260804-continuous-openui-schedu-1e62ecf9-c{3,4,5}/`
(local, explicit no-sync); prior docs
[c3](autotrain-cycle-continuous-openui-scheduled-c3-decode-timeout-diagnostic.md),
[c4](autotrain-cycle-continuous-openui-scheduled-c4-decode-timeout-confirmed.md).
