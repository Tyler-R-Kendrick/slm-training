# Iteration: eight-step loss versus generation feedback (2026-07-15)

Seed 0, batch size 8, learning rate `6e-4`, and final-only feedback reached
weighted held-out NLL **17.410** at **11,122** target tokens after eight
steps. Telemetry and loss/insight artifacts were complete; effective batch
size was recorded as 8.

A bounded unconstrained smoke evaluation of the resulting checkpoint did
complete and persisted AgentV artifacts, but parse rate, structural similarity,
and reward were all **0**. The constrained path remains too slow to score.
Therefore the improved teacher-forced loss is not evidence of generation
readiness, and the checkpoint is not promoted or treated as a ship result.
