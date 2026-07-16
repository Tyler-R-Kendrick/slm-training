# E106 repair mixture with quality threshold (2026-07-15)

E106 added and exercised train-time quality filtering for online mixture
sampling. The repair-heavy E103 mixture was retained, but records below
quality score `0.95` were excluded before family/task pools were built. This
reduced the training population from 1,417 to 1,303 records.

The 128-step CPU run completed with loss `7.63232`, 39,597 target tokens, and
persisted telemetry including the mixture threshold and filtered count. Strict
smoke evaluation remained invalid: parse/raw syntax `0.0/0.0`, structural
similarity `0.1167`, contract precision/recall `1.0/1.0`, placeholder fidelity
`1.0`, component recall `0.0`, and latency `13520.38 ms`. AgentV remained
non-ship with 5 failed checks.

Decision: reject the quality-threshold repair mixture as a model candidate.
Retain the harness capability because it enforces the judge gate at sampling
time; the result shows that repair distribution, not the low-score tail alone,
is driving the failure.
