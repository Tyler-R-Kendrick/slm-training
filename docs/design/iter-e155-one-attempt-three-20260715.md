# E155 — One-attempt three-record validation (2026-07-15)

The one-attempt policy was evaluated on three smoke records with the E147 checkpoint. It produced no timeouts, p50 latency `7,551.3 ms`, p95 latency `10,390.0 ms`, structural similarity `0.2067`, and placeholder validity `0.2667`. Parse remained `0.0` on all three records and AgentEvals remained `0/5`. This validates the retry cap as a latency guard, not as a quality fix.
