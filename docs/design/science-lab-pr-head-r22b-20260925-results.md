# R22b PR-head diagnostic — incomplete, 25 September 2026

[Machine-readable measured record](science-lab-pr-head-r22b-20260925-results.json). Campaign `science-lab-pr1785-0ea5ba4-r22b-20260925` used immutable source commit `0ea5ba480a138883ec43eda57781bb7c39bb1262` (tree `a227bf1ea06f23a03f80a43d1ae3771e5fa6a28d`, marker digest `1b50be5f5a6a688e58c792fcbaebf80287ab0850a0cc7bd6abdb62f1b2b55e09`). Preregistration SHA-256 `f53f18ec2458a891722244ef26a0b2ff687f39ad8af6a817fa56c6d88faa0ffd` locked control manifest `deee770df944d7fb59692c7dd2908f74a049a247562a67da0e5e0a440b08ed30` and candidate manifest `13d247ce59fdacf410ec5662ba38ee0415b88119626e99f9d2fcd4d06a2b41df`. Label: **PR-head diagnostic**; `promotion_allowed=false`.

## Recipe and outcome

CPU scratch TwoTower; seed 7301; eight training records; batch size 2; 65,826 parameters per arm; three-update ancestor; six recorded logical steps in chunks of three; control learning rate 0.0003, candidate 0.0006. Planned endpoint: six selected public smoke cases, paired `smoke.eval_nll` decrease of at least 0.02 nats/token. Selection digest: `11ccffe86a6cf32ed1788721a39e6f1818fb91a0b9cedd0560a8ee4c3f29c3ff`. Execution followed ordinary supervisor and driver, with local `--no-sync-checkpoints` artifacts.

| Arm | Training | Local checkpoint SHA-256 | Smoke decode | Available descriptive smoke metrics | Outcome |
| --- | --- | --- | --- | --- | --- |
| Control | 6 steps completed; final training loss 40.569138 | `673fce9d06eee0b226d6520c1a8f32c9dacbc924ac8997d15a5e765df2b465b8` | 2/6 complete; 4 pending | Parse 1.0, meaningful-program rate 0.5, structural similarity 0.35, reward 0.0 **on two decoded cases only** | Eval continuation exit 2: `evaluate_model.py: error: argument --resume-run: expected one argument` |
| Candidate | 6 steps completed; final training loss 37.893265 | `0a125ff5e1ebcf4b456e8f0d432ca03a187eb298f045b77b8a13c81e62d27313` | 0/6 complete; 6 pending | Unavailable | Eval yielded exit 10 with records pending |

Both smoke scoreboards report `measurement_complete=false` and `publication_complete=false`. Training loss is not paired evaluation NLL. Control `eval_nll_records.json` covers 96 records under a different selection, so it cannot substitute for preregistered six-case endpoint. **Control NLL, candidate NLL, paired effect, and endpoint verdict remain unavailable.** No completed six-case AgentV SDK bundle or ship-gate evidence; no promotion, shipment, or main-delivery authority.

## Evidence and version

- Preregistration: `outputs/autonomy-integration-20260921/science-lab-r22b-0ea5ba4-prereg/preregistration.json`.
- Campaign events: `outputs/autonomy-integration-20260921/science-lab-r22b-0ea5ba4-prereg/campaigns/science-lab-pr1785-0ea5ba4-r22b-20260925/events.jsonl`; supervisor log: `outputs/autonomy-integration-20260921/science-lab-r22b-0ea5ba4-prereg/campaigns/loops/science-lab-pr1785-0ea5ba4-r22b-20260925/supervisor.jsonl`.
- Control run: `outputs/autonomy-integration-20260921/science-lab-r22b-0ea5ba4-prereg/campaigns/science-lab-pr1785-0ea5ba4-r22b-20260925/runs/science-lab-pr1785-0ea5ba4-r22b-20260925-control/` — `train_summary.json`, `checkpoints/last.pt`, `eval_smoke.partial.json`, `scoreboard.json`.
- Candidate run: `outputs/autonomy-integration-20260921/science-lab-r22b-0ea5ba4-prereg/campaigns/science-lab-pr1785-0ea5ba4-r22b-20260925/runs/science-lab-pr1785-0ea5ba4-r22b-20260925-candidate/` — same artifact names.
- Eval version stamp (`version_stamp/v1`): code `0ea5ba480a138883ec43eda57781bb7c39bb1262`, `code_dirty=false`; `harness.model_build.eval=v107`, `evals.meaningful_program=2.14.0`, `gates.ship` absent from incomplete scoreboard stamp. Full component map in JSON.

Harness failure requires source repair and fresh source-bound evidence before any comparison claim. Frozen R22b campaign remains incomplete.
