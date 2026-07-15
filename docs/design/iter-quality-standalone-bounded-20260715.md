# Iteration: bounded standalone generated evaluation (2026-07-15)

The standalone evaluator now accepts both `--eval-limit` (records per suite)
and `--gen-steps` (decode steps). Defaults preserve full evaluation behavior;
the flags are explicitly diagnostic controls.

Two one-record smoke probes against the four-step batch-8 scratch checkpoint
were attempted. The first used the record cap with the default eight decode
steps; the second used `--gen-steps 1`. Both were terminated by the execution
environment during decoding before a scoreboard was written. Therefore no
generated-quality metric or ship conclusion is claimed. The loss feedback path
remains complete and authoritative for these runs.

A third isolation probe removed DESIGN context and structural trust, then set
`--max-attempts 1`; it was still terminated before a scoreboard. This rules out
the obvious retry/context multipliers and leaves checkpoint-load/model-decode
profiling as the next required diagnostic. No generated-quality claim is made.

Stack tracing identified `openui_langcore.stream_check` inside constrained
token selection. The CLI now exposes `--skip-exact-stream-probe`. A retry with
that flag, one decode step, one attempt, and a toy grammar backend still failed
to produce a scoreboard, so checkpoint/model initialization or another decode
path remains to be isolated.

The skip setting was also wired through MaskGIT remask verification, where
`filter_ids_by_stream` could re-enter LangCore after token selection. The
bounded constrained retry still exceeded the environment window, so this is a
performance intervention with regression coverage, not a quality result.

The unconstrained model-forward control did complete: one-record smoke,
one-step, one-attempt evaluation finished in **3.39 s** and persisted AgentV
artifacts. It produced parse rate **0**, structural similarity **0**, and reward
**0**. This is diagnostic feedback only, but it localizes the interruption to
the constrained-decoding path rather than checkpoint loading or evaluator
persistence.

The constrained path was also tested after making `skip_exact_stream_probe`
skip the redundant post-DFA stream probe and disabling structural preference;
it still exceeded the execution window. Grammar/DFA regression tests passed
(31 tests), but no constrained quality metric is claimed until the remaining
stream-check call site is isolated.
