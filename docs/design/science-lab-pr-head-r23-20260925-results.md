# R23 PR-head diagnostic — control complete, pair incomplete

[Machine-readable record](science-lab-pr-head-r23-20260925-results.json). Source commit `cf5432c91a2f2c19005e6c7c74f7f63ee858e4cd`; tree `16c290d01502c6d3c73aae51180ad397e825d674`; authenticated execution marker digest `773b9a24c75f6f8ba928385bf7ee524a7b1c99a4a3191528873defb06a978dcd`. Campaign `science-lab-pr1785-cf5432-r23-20260925`; preregistration SHA-256 `73f12a41926e54f0521fed4c730c029ca9f5f5e58a772b8302e4d9c1b93918bc`; control/candidate manifest digests `8fd90bb311f0ca4a31ff346b2a591700d3e5d77b9cd3506e1ffa3f1aff45cd29` / `982c06c8f4ff8e295937cff0efc2a0dab8c574f3e240ea62b5aec9fdfca480bd`. Label **PR-head diagnostic**, `promotion_allowed=false`.

## Recipe and measured control

CPU scratch TwoTower; seed 7301; batch 2; eight training records; 65,826 parameters per arm; three-update ancestor; six recorded steps in chunks of three. Control LR 0.0003; candidate LR 0.0006. Planned endpoint: paired six-case `smoke.eval_nll` decrease ≥0.02 nats/token. Locked selection digest `11ccffe86a6cf32ed1788721a39e6f1818fb91a0b9cedd0560a8ee4c3f29c3ff`. Local checkpoints use `--no-sync-checkpoints`.

Control `science-lab-pr1785-cf5432-r23-20260925-control` completed training (`steps=6`, final training loss `40.569138`), wrote checkpoint `outputs/autonomy-integration-20260921/science-lab-r23-cf5432-prereg/campaigns/science-lab-pr1785-cf5432-r23-20260925/runs/science-lab-pr1785-cf5432-r23-20260925-control/checkpoints/last.pt` (SHA-256 `673fce9d06eee0b226d6520c1a8f32c9dacbc924ac8997d15a5e765df2b465b8`), and completed six of six public smoke decodes. AgentEvals JSONL and AgentV SDK bundle exist. Six-case metrics from `eval_smoke.json`:

| Metric | Control |
| --- | ---: |
| Parse rate | 1.000000 |
| Meaningful-program rate | 0.166667 |
| Structural similarity | 0.244450 |
| Component-type recall | 0.125000 |
| Binder-reference F1 | 0.944444 |
| Placeholder fidelity | 0.916667 |
| AST / canonical BEq | 0.000000 / 0.000000 |
| Reward | 0.000000 |
| Median latency | 768.42 ms |

`gates.json` reports **fail**, 11 failures: smoke n=6 below 20, meaningful-program rate, structure, component recall, AST/canonical BEq, reward, plus missing held-out, adversarial, OOD, and rico-held suites. AgentV evidence is control-only. Training loss is not evaluation NLL. Control `eval_nll_records.json` covers 96 records under different selection; its mean does not certify locked six-case endpoint.

## Incomplete pair and retained journal

Candidate experiment and command cursor started. No candidate run directory, checkpoint, train summary, evaluation, or outcome exists. Outer supervisor timeout ended the invocation during candidate cursor; retained driver state and supervisor log show `waiting_capability` for same cursor. **No candidate metrics, paired NLL, measured effect, endpoint verdict, promotion, or shipment claim.** Journal stays intact. The unknown started cursor cannot be replayed without independent proof; corrected source requires a fresh locked campaign.

Evidence: `outputs/autonomy-integration-20260921/science-lab-r23-cf5432-prereg/campaigns/science-lab-pr1785-cf5432-r23-20260925/events.jsonl`; `outputs/autonomy-integration-20260921/science-lab-r23-cf5432-prereg/campaigns/loops/science-lab-pr1785-cf5432-r23-20260925/supervisor.jsonl`; `outputs/autonomy-integration-20260921/science-lab-r23-cf5432-prereg/campaigns/science-lab-pr1785-cf5432-r23-20260925/artifacts/driver_cycle_state/08e48ae334901913e309a9c9bb039176c0bcbbc4e96e1b268d35d19f54b367a8.json`. Control scoreboard: `outputs/autonomy-integration-20260921/science-lab-r23-cf5432-prereg/campaigns/science-lab-pr1785-cf5432-r23-20260925/runs/science-lab-pr1785-cf5432-r23-20260925-control/scoreboard.json`; AgentEvals JSONL: `outputs/autonomy-integration-20260921/science-lab-r23-cf5432-prereg/campaigns/science-lab-pr1785-cf5432-r23-20260925/runs/science-lab-pr1785-cf5432-r23-20260925-control/agentv/openui-model-ship-gates-2026-09-25t18-17-44-568018-00-00.eval.jsonl`; AgentV bundle: `outputs/autonomy-integration-20260921/science-lab-r23-cf5432-prereg/campaigns/science-lab-pr1785-cf5432-r23-20260925/runs/science-lab-pr1785-cf5432-r23-20260925-control/agentv/openui-model-ship-gates-2026-09-25t18-17-44-568018-00-00`.

Eval `version_stamp/v1`: code `cf5432c91a2f2c19005e6c7c74f7f63ee858e4cd`, `code_dirty=false`, `harness.model_build.eval=v107`, `evals.meaningful_program=2.14.0`. Ship-gate version `openui_ship_gates_v6`. Full stamps in JSON.
