# autotrain_wf_smoke_20260725_iter281

**Honesty:** fixture_or_scratch. **Not ship.**

train_version=wf_smoke_v2 last_loss=35.92511749267578 stopped_on=steps wall=53.291296923 max_wall=2.5833333333333335 n=3

Fixture note: `dq_strict_fixture_r4_20260718` now fails
`assert_symbol_only_output` (free-form string `'contact'`) under the current
`symbol_only/v2` output contract, so this iteration trained against the
committed symbol-only-compliant fixture
`src/slm_training/resources/data/train/e1291_document_only_full_program_v1`
(350 records) instead — real `slm sft train` / `slm eval model` run, same
`--fast-train --no-sync-checkpoints --steps 8 --model twotower --device cpu`
recipe, `--eval-limit 3 --suites smoke --run-class fixture_demo`. AgentV
criteria: pass=false, passed=2/7 (as expected for a 3-record diagnostic
subset; not a ship claim).
