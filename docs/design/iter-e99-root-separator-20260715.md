# E99 root separator diagnostic (2026-07-15)

E99 forced a newline after a completed `root = ...)` expression before repair
statements. The rule removed no dead ends, but made the strict smoke decode
materially worse: structural similarity fell to `0.3458` and latency rose to
`58239.16 ms` (E98: `0.5333` and `9205.65 ms`). Root bypass telemetry remained
`2` and constrained dead ends remained `0`.

Decision: revert the separator force. The grammar must not synthesize a
statement boundary until the LTR repair path can certify it cheaply. Keep the
root admission fix and investigate repair-state training/transition handling.
