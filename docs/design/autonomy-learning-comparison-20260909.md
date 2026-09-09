# A12 learning comparison — 2026-09-09

The canonical `slm experiments learning-comparison prepare` entrypoint ran on
`autonomy-a12-fixture-20260909b` with the explicit repository bridge override.
It completed on CPU with one update and emitted choice diagnostics, learner
observations, manifests, quality reports, and a local ancestor checkpoint.

The first prepare invocation failed because the candidate bridge dependency was
not resolved; the rerun with
`OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs`
completed. The subsequent bounded continuation completed both matched arms,
decoded probes, and AgentV publication with zero SDK execution errors. The
locked paired masked-denoising NLL diagnostic is complete but inconclusive
(n=2, n_nontied=2, mean delta -0.0034681 nats/masked token, Wilcoxon p=1.0,
insufficient-nontied-pairs reason). This is conditional fixture evidence, not
an independent production confirmation; its fixture gate failed and no
checkpoint was promoted or synced.

Evidence: [machine-readable result](autonomy-learning-comparison-20260909.json)
and the run root under `outputs/runs/autonomy-a12-fixture-20260909b/`.
