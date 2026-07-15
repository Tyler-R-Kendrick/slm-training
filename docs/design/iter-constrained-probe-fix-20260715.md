# Iteration: constrained fallback probe fix (2026-07-15)

The constrained decoder had a duplicated semantic stream probe in its fallback
candidate-ranking loop. Although `skip_exact_stream_probe` was enabled, the
loop called `stream_check()` directly for every candidate after `_legal()` had
already admitted it. On CPU this could make a one-record, one-step evaluation
appear hung.

The fallback now honors the same probe policy as `_legal()`. A bounded
constrained smoke evaluation of the seed-2 eight-step checkpoint completed in
**2,671.66 ms**, persisted a scoreboard and AgentV JSONL, and recorded
`decode_timeout_count=0`. Before the fix, equivalent constrained runs were
stopped before producing a scoreboard.

Quality remains unchanged and invalid: parse rate **0.00**, reward **0.00**,
and structural similarity **0.30**. This validates termination and
observability only; it is not generation readiness or a ship claim.

Recipe: scratch CPU checkpoint, one smoke record, one generation step, one
attempt, `skip_exact_stream_probe=true`, `verify_chosen_only=true`, and a five
second decode timeout.
