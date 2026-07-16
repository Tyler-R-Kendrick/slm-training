# Iteration: decode timeout telemetry (2026-07-15)

The standalone evaluator now has an explicit per-record decode timeout for
diagnostic runs. The timeout control was verified on the unconstrained
checkpoint path: it completed a bounded scoreboard and persisted AgentV
artifacts without changing the default unlimited behavior.

The constrained Lark path remains different: its blocking execution does not
yield to the Python signal handler, so `decode_timeout_count` cannot yet
interrupt that path. The stack is in Lark rather than the OpenUI bridge. A
subprocess-isolated grammar worker is the next safe design if constrained
timeouts must be enforced. No generated-quality or ship claim is made.
