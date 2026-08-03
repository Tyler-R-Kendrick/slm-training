# Autotrain c2 (continuous-openui-local): AgentV SDK missing, then NODE_OPTIONS, not a model result

**Verdict:** infrastructure failure, not scoreable. Training completed for both
the control and `-bounds` `wf_smoke_v2` arms (1,608,962 params, 21 steps,
losses `22.6219` for both, checkpoints `f2fe8f5a...b81b2` control /
`2c1749c6...41742f64` bounds, local explicit no-sync), but
`evaluate_model.py --ship-gates` crashed before producing a scoreboard for
either arm: `RuntimeError: AgentV SDK is unavailable; run npm ci in the
checkout or set AGENTV_RUNNER`. Neither arm has smoke metrics; this is not
evidence about the model.

Root cause and fix, in two layers (commits `1faeff44` and `8da7b777`):

1. This sandbox's checkout never ran `npm ci` at the repo root, so
   `node_modules/@agentv/core` didn't exist. Added it to
   `scripts/setup_dev_env.sh`.
2. Installing the SDK exposed a second, more general bug while investigating:
   the sandbox exports `NODE_OPTIONS="--import tsx" ...`, which this Node
   build rejects with exit 9 and empty stdout — the same failure mode
   `src/slm_training/dsl/grammar/backends/graphql_js.py` had already worked
   around locally, but `src/slm_training/dsl/lang_core.py` (the OpenUI
   lang-core bridge behind every `validate()`/`parse()` call, including
   ProgramSpec generation's G2 schema gate) and `src/slm_training/evals/agentv.py`
   (the AgentV runner subprocess) had not. Generalized the fix into
   `slm_training.bridge_utils.sanitized_node_env()` and wired it into all
   three node/npm subprocess call sites, each with a regression test
   asserting the subprocess receives a cleared `NODE_OPTIONS`.

No scoreboard, no smoke metrics, no ship-gate result exists for this cycle;
the checkpoints are local, explicit no-sync, and not reusable, promotable, or
ship evidence. Lean is `not_applicable:screening`.

Next: replay the identical frozen `-bounds` arm (`retry_measurement`) now that
training, evaluation dependencies, and the Node bridge subprocess environment
are all sound in this sandbox.

Machine evidence:
[`autotrain-cycle-c2-agentv-missing-infra-failure.json`](autotrain-cycle-c2-agentv-missing-infra-failure.json).
