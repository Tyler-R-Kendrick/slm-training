# autotrain_wf_smoke_20260725_iter283

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=35.92511749267578 stopped_on=steps wall=45.768244281 max_wall=2.5833333333333335 n=3

Same recipe/fixture as iter281/282 (`e1291_document_only_full_program_v1`,
`--fast-train --no-sync-checkpoints --steps 8 --model twotower --device cpu`,
`--eval-limit 3 --suites smoke --run-class fixture_demo`); deterministic
re-run (no `--seed` override) so last_loss again matches. AgentV criteria:
pass=false, passed 1/7.
