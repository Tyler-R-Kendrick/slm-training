# Autotrain loop ledger (fixture smoke) — 2026-07-26

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

Continuation of
[`autotrain-loop-ledger-20260725.md`](autotrain-loop-ledger-20260725.md)
(iter1–iter50). iter51 first reproduced a real regression — see
[`autotrain-smoke-canonical-marker-regression-20260726.md`](autotrain-smoke-canonical-marker-regression-20260726.md)
— fixed it (`harness.train_data` v20 → v21), then re-ran the smoke recipe
end to end.

| run_id | ok | steps | stopped_on | last_loss | wall_s (train) | max_wall | mpr | dtc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `autotrain_wf_smoke_20260726` (iter51) | True | 8 | steps | 30.21893310546875 | 6.57 | 2.5833333333333335 | 0.0 | 3 |
| `autotrain_wf_smoke_20260726_iter52` | True | 8 | steps | 33.309722900390625 | 2.93 | 2.5833333333333335 | 0.0 | 0 |
| `autotrain_wf_smoke_20260726_iter53` | True | 8 | steps | 27.57246208190918 | 2.71 | 2.5833333333333335 | 0.0 | 3 |
| `autotrain_wf_smoke_20260726_iter54` | True | 8 | steps | 32.53175354003906 | 2.66 | 2.5833333333333335 | 0.0 | 0 |
| `autotrain_wf_smoke_20260726_iter55` | True | 8 | steps | 35.857765197753906 | 2.31 | 2.5833333333333335 | 0.0 | 3 |

`mpr` = `meaningful_program_rate`, `dtc` = `decode_timeout_count` (both from
the AgentV smoke suite scoreboard). `meaningful_program_rate` is expected to
stay `0.0` at 8 scratch steps — this recipe verifies pipeline wiring, not
model quality; see `honest-ship-eval`.
