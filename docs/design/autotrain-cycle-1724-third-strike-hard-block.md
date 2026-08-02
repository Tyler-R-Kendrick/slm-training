# Autotrain c1724: third consecutive decode timeout — hard block

**Verdict:** three consecutive cycles (c1722, c1723, c1724) have now failed
honest smoke evaluation on both arms with a wall-time interrupt inside the
same `completion_kernel._eval` recursion family, each time in a different
child frame but the same top-level call chain
(`build_completion_forest -> terminal_witness -> completion_kernel._eval ->
advance_path -> advance -> engine snapshot/copy`). Per the continuous-loop
absolute law ("stop only when blocked... same hard blocker has failed three
consecutive cycles with no new information"), this session's autotrain loop
stops here rather than spending a fourth blind retry.

## Result matrix

| Cycle | Arm | Checkpoint source | Interrupt frame | Outcome |
| --- | --- | --- | --- | --- |
| c1722 | control, canvas | fresh training this cycle | `semantic_state.advance -> dataclasses.replace` | timeout |
| c1723 | control, canvas | c1722 checkpoints (reused) | `engine._ip_control_copy -> copy.copy` (from `advance`) | timeout |
| c1724 | control, canvas | c1722 checkpoints (reused) | `engine._ip_control_copy -> copy.copy` (from `_feed_terminal_direct -> _snapshot`) | timeout |

Full stderr traces and frozen manifest digests are in the
[machine-readable record](autotrain-cycle-1724-third-strike-hard-block.json).
c1721 (this session's first cycle) completed both arms' full train+eval
within budget, so the harness itself is not universally too slow in this
container — only decode under this specific checkpoint/manifest lineage has
now failed three times running.

## Why this is a hard block, not another soft-fail retry

- Same evidence class each time: an `AgentEvals`/`evaluate_model` subprocess
  interrupted by the driver's own cooperative wall-time enforcement, zero
  scoreable metrics, no exception other than the interrupt.
- Same top-level call chain each time — real signal, not noise — but the
  exact child frame differs cycle to cycle, meaning the recursion has no
  single obviously-slow leaf; naming a "fix" from any one trace would be a
  guess, which c1719's playbook and this doc's own predecessors (c1722,
  c1723) already ruled out as the next step without call-count evidence.
- Retrying a fourth time with the same frozen manifest and no new
  instrumentation would not produce new information — it is expected to
  reproduce the same class of timeout.

This container provides 4 CPU cores, no GPU, and the repo-wide
`MAX_RUN_MINUTES=3` wall cap (`src/slm_training/levers.py`) is
non-negotiable and was not modified or bypassed. Whether the decode path is
inherently over budget on this specific model/checkpoint combination on this
class of hardware, or a real algorithmic regression exists, is not yet
distinguishable from the evidence collected so far.

## Required next step (not undertaken this session)

Per the c1719 counted-probe playbook: instrument
`completion_kernel._eval`/`advance`/`engine.copy_control` with interrupt-safe
partial counters (calls per frame, cumulative wall time per frame) and
replay the identical c1722 frozen manifest
(`b1ec053df914d11a5648aa57ba0ea8ca5098a3cc8c399bc58fe872c5dc1ccd4f`) once
with that instrumentation attached. That is a `improve-openui-harnesses`
task requiring careful review (adding counters must not change grammar
membership, scores, or fail-closed validation), not a blind scheduling
tweak, so it is left for a follow-up session/PR rather than guessed here.

## Session closeout

No positive-result cycle occurred this session (c1721–c1724 are all
`SDLC_PHASE_A NON_POSITIVE`), so per `autotrain-iteration-delivery` no stack
layer/PR was opened at any point — all four cycles are local commits only,
pushed to `claude/great-dirac-egbbjc`. No checkpoint promotion, no theorem
optimum, no model-card change. This doc plus c1721–c1723 are the durable
record; the next session should start with the counted-probe instrumentation
above before attempting another blind frozen replay.

No component version bump — this cycle changed no versioned file.
