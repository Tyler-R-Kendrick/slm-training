# E140 E135 checkpoint replay and evaluation-policy audit — 2026-07-15

E140 replayed the exact E135 checkpoint twice with the same diagnostic policy,
then once with the checkpoint's stored verification policy.

| Policy | Verify chosen only | Attempts | Parse | Placeholder validity | Structure | Timeouts | p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Current diagnostic | yes | 1 | 0.0 | 0.4500 | 0.3311 | 0 | 18,945 ms |
| Current diagnostic repeat | yes | 1 | 0.0 | 0.4500 | 0.3311 | 0 | 18,897 ms |
| Checkpoint-stored policy | no | 3 | 0.0 | 0.1333 | 0.1189 | 2 | 20,001 ms |

The fixed-policy replays are deterministic. The policy change materially alters
quality and timeout metrics even though the checkpoint SHA is identical. This
explains why the historical E135 result cannot be compared directly with E138
and E139. All policies still fail parse and reward gates.

The evaluator now persists `evaluation_policy` in every scoreboard, including
context backend, constrained mode, probe policy, chosen-token verification,
attempt count, timeout, fallback policy, and generation limits. Future training
feedback must use one declared policy per comparison family.
