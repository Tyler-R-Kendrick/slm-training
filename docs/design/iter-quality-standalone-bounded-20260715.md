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
