# Continuous autotrain cycle 2 results (2026-08-02, `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2` |
| Source | `5ff93f378a68adafd0da8442549501b3e7ccda41` |
| Cycle intent | `retry_measurement` (frozen replay of cycle 1's `c1-control`/`c1-bounds`) |
| Train | `wf_smoke_v2` (reused checkpoints, `FROZEN_TRAIN_REUSE`) |
| Eval | `e938_role_safe_all_targets_v2` |

## What happened

This cycle replayed cycle 1's frozen control/bounds training arms (reusing
their checkpoints rather than retraining) to complete the measurement that
cycle 1 left incomplete. Both arms crashed again at evaluation, this time
with a more specific error now that `npm ci` had installed `@agentv/core`:

```
RuntimeError: AgentV SDK evaluation failed: node: --import tsx is not allowed in NODE_OPTIONS
```

`publish_agentv_evaluation` (`src/slm_training/evals/agentv.py`) spawns
`node scripts/run_agentv_eval.mjs` and inherited the parent process's full
environment, including a host `NODE_OPTIONS` carrying an unrelated
`--import tsx` loader flag. Node refuses to start under that combination,
so the AgentV bundle publish step — and therefore the whole
`evaluate_model.py --ship-gates` invocation — crashed after the core
`eval.json` scoreboard had already been written.

This was the second identical-arm replay failure, exhausting the
consecutive-incomplete-replay budget and raising a typed `repair_harness`
action (`harness_family: model_build`).

## Repair

Fixed in `14ddf5e2e8a15e9056171d49e5e57f9e8c7a0020`: strip `NODE_OPTIONS`
from the subprocess environment before spawning the AgentV node runner —
the runner is a plain ESM script and needs none of the host's loader flags.
The same commit refreshed two `test_evals/test_agentv.py` fixtures that
predated `meaningful_program_rate`/`ast_beq_rate`/`canonical_beq_rate`
joining the canonical smoke ship-gate policy (they were silently exercising
a partial gate set).

## Next-run priorities

1. **infrastructure:** replay the identical frozen `c2-control`/`c2-bounds`
   arms now that the harness fix is in place.
2. **model:** re-test `grammar_completion_bounds` for a real
   latency/quality delta only once the replay produces a complete
   AgentV-backed scoreboard.
3. **evaluation:** keep ship gates honest; do not promote/sync/ship either
   fixture checkpoint.

## Artifacts

- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c2/`
- Repair commit: `14ddf5e2e8a15e9056171d49e5e57f9e8c7a0020`
- JSON twin: `continuous-openui-20260802-c2-results.json`
