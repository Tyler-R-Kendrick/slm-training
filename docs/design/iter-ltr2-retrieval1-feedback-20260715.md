# Iteration: LTR2 retrieval-1 feedback (2026-07-15)

A 64-step seed-0 LTR2 run tested one nearest skeleton retrieval (`retrieval_k=1`) against the unchanged remediated corpus. Interval held-out loss feedback ran every 32 steps and the bounded smoke probe evaluated one record with AgentV persistence.

Held-out weighted NLL was **7.382**. Bounded smoke feedback produced **0.425 structural similarity**, **0.1333 placeholder validity**, and **0.500 component recall** at **3.32 s** p50 latency. Parse rate and reward remained **0**, and AgentV recorded 0 passed / 1 failed.

The matched LTR2 control was 0.5375 structural similarity and 0.2 placeholder validity with the same zero parse and reward rates. Reject retrieval-1 for this branch; the next intervention should target serialization/token supervision rather than adding more contextual scaffolding. Full train, scoreboard, AgentEvals, and AgentV artifacts remain under the run directories.
