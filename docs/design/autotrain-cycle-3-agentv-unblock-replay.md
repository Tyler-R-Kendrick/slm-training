# Autotrain c3: AgentV unblock, replay-proven

**Verdict:** positive result via **executable unblocking** (not a model
quality win). Cycle 3 of loop `continuous-openui-20260802-local` replays the
identical frozen c2 manifests. Both arms now complete end-to-end with real,
honest scoreboards instead of crashing.

## Why this is positive

`autotrain-iteration-delivery.md` defines three positive-result categories.
This cycle satisfies category 3:

> Executable unblocking — a harness/path/code fix removes a prior hard path
> error or unrecoverable blocker and the identical arm then completes with a
> usable scoreboard (replay-proven). Knob thrash that still fails the same
> way is not positive.

- c2 (frozen source): both arms crashed with an unrecoverable
  `RuntimeError` (AgentV SDK unavailable, then a `NODE_OPTIONS` rejection —
  see [c2 doc](autotrain-cycle-2-agentv-node-options-gap.md)).
- Fixes applied: `npm ci` (local) + commit `ca15f5c` (`_sanitized_env()` in
  `src/slm_training/evals/agentv.py`, `evals.agentv` v6→v7).
- c3 (this cycle): the **identical frozen manifests**
  (`c20260802-continuous-openui-202608-75a7803e-c3-{control,bounds}`, same
  reused checkpoint, same recipe) complete both training and evaluation and
  produce real `scoreboard.json` / `gates.json` artifacts.

The driver's automatic `SDLC_PHASE_A` classifier tagged this cycle
`NON_POSITIVE` (`fixture_insufficient_n`,
`primary_metric_null_or_worse:...improvement=0.0`) because it only checks
primary-metric deltas and ship-quality wins on a comparative arm — it has no
signal for "a prior hard crash now completes." That check is correct on its
own terms (there is no model-quality delta here; control and candidate are
the same reused checkpoint) but it is not the full picture: this cycle is
the agent-judged replay-proof required to close out the c1/c2 infrastructure
repair as positive, per the harness family's `improve-openui-harnesses`
contract ("replay the identical frozen arm" after a `repair_harness`
action).

## Result matrix

| Arm | Suite n | parse_rate | binder_reference_f1 | meaningful_program_rate | Ship gates | Honest disposition |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| control | 3 | 1.0 | 0.6333 | 0.0 | fail (`insufficient_n`, 6 quality thresholds, 4 missing suites) | Expected fixture rejection |
| bounds | 3 | 1.0 | 0.6333 | 0.0 | fail, identical to control | Expected fixture rejection |

`control == candidate` (both reuse the c2 checkpoint; `improvement=0.0`), so
there is **no primary-metric claim** here — only that both arms are now
scoreable at all. The ship-gate rejection itself is correct and expected for
`n=3` smoke-scale fixture data against a 20-document minimum; it is not an
infrastructure failure.

## Delivery

This closes the positive layer opened in
[PR #1292](https://github.com/Tyler-R-Kendrick/slm-training/pull/1292) on
`claude/great-dirac-ptxx92`: `torch` install docs (c1) → AgentV
`NODE_OPTIONS` fix + stale-fixture repair (`ca15f5c`) → this replay-proof
(c3). No checkpoint was promoted (ship gates correctly reject), so no
`MODEL_CARD.md` update applies.

Eval commit: `a9fcdf30604ac87cd69382705670db03c46629ab`
(`evals.agentv=v7`).
