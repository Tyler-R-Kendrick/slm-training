# A12 learning comparison — 2026-09-09

The canonical `slm experiments learning-comparison prepare` entrypoint ran on
`autonomy-a12-fixture-20260909b` with the explicit repository bridge override.
It completed on CPU with one update and emitted choice diagnostics, learner
observations, manifests, quality reports, and a local ancestor checkpoint.

The first prepare invocation failed because the candidate bridge dependency was
not resolved; the rerun with
`OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs`
completed. The subsequent corrective comparison did not produce a complete
paired result within the bounded invocation. This is an operationally
incomplete fixture, not a negative model result, and it is not AgentV/ship or
champion evidence. No checkpoint was synced or promoted.

Evidence: [machine-readable result](autonomy-learning-comparison-20260909.json)
and the run root under `outputs/runs/autonomy-a12-fixture-20260909b/`.
