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
