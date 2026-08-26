# Continuous cycle `continuous-loop-20260826-continuous-openui-local-8c0b60dd-c561`

- loop_id: `continuous-openui-local`
- cycle_index: `561`
- role/intent: `screening` / `screening`
- primary_metric: `smoke.eval_nll`
- measurement_complete: `False`
- evidence_class: `fixture`
- result: fail-closed before either arm started
- reason: `symmetric decision-arm budget` had 146.421 seconds remaining for two frozen 70-second arms plus finalization
- arm scoreboards: none

## Harness follow-up

- Cycle preflight recovered another 13.5 seconds after the persistent lineage cache, leaving an 8.8-second shortfall.
- Profiling isolated three fresh Python processes paying the package root's eager DSL import cost.
- Package-root legacy DSL exports now load on demand: plain startup measured 0.02 seconds versus 3.57-3.85 seconds before the repair; explicitly importing the legacy exports measured 2.50 seconds.
- One subprocess regression check preserves `from slm_training import ExampleRecord, parse` while proving plain startup does not import `slm_training.dsl`.
- The first supervised restart failed before campaign creation because the new import order exposed an eager `dsl.production_codec` / `data.contract` cycle; no c562 cycle was initialized.
- Production-codec package exports now load only when requested. The regression check reproduces the supervisor order (`data.store` first) and then verifies both legacy root and production-codec exports.
- Focused tests and Ruff passed; normalized version-stamp and repository-policy results are recorded with the repair commit.

Partial fixture orchestration only. No train, eval, checkpoint, AgentEvals result, or ship claim was produced.
