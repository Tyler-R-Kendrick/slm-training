# R24 PR-head diagnostic — control complete, candidate incomplete

[Machine-readable record](science-lab-pr-head-r24-20260925-results.json). Source commit `e9a6de0c6c52e4326252e2c6c76d1abad423d690`, tree `03e99f302ed5ad440f2d740b3c0a03d7664ce6b2`, authenticated execution marker digest `4bfc1dc871e043f08e547d44f9ee6d3adbbc1b6a8916f180291cccd63b57d458`. Campaign `science-lab-pr1785-e9a6de-r24-20260925`; preregistration SHA-256 `3f051cd4559451a3e7a68b9ebd0412cb2b2126f71b61fcf9bcee4f62971c7d01`; control/candidate manifest digests `250de7f88c822e525df46e58f412982f6b4c30e3b7875a87ede4feb7d60f42da` / `cfac6769d833b86d063367ec55fcc4a963a069b0150f62ce089dd11def271be9`. Label **PR-head diagnostic**, `promotion_allowed=false`.

## Recipe and measured control

CPU scratch TwoTower; seed 7301; batch 2; eight training records; 65,826 parameters per arm; three-update ancestor; six recorded steps in chunks of three. Control LR 0.0003; candidate LR 0.0006. Planned endpoint: paired six-case `smoke.eval_nll` decrease ≥0.02 nats/token. Local checkpoints use `--no-sync-checkpoints`.

Control `science-lab-pr1785-e9a6de-r24-20260925-control` completed six training steps and six of six public smoke decodes. It published AgentEvals JSONL and an AgentV SDK bundle with zero SDK execution errors. Control smoke metrics: parse rate 1.000000, meaningful-program rate 0.166667, structural similarity 0.244450, component-type recall 0.125000, binder-reference F1 0.944444, placeholder fidelity 0.916667, AST/canonical BEq 0/0, reward 0, and median latency 767.89 ms. AgentV passed 2 of 9 criteria; ship gates failed 11, including smoke sample volume and missing held-out, adversarial, OOD, and `rico_held` suites. This is diagnostic evidence only.

## Incomplete candidate and retained evidence

Candidate `science-lab-pr1785-e9a6de-r24-20260925-candidate` completed six CPU training steps and wrote a 65,826-parameter local checkpoint. Its ordinary-supervisor smoke evaluation timed out with exit 124 after a command cursor had started. No candidate scoreboard, AgentV result, terminal `experiment_finished` event, or paired NLL exists. The command cursor journal records an unknown started evaluation; it must not be counted as a completed measurement. **No paired verdict, promotion, champion, or shipment claim.**

The controller later reconciled the partial control evaluation through its fenced cursor and completed control measurement. The candidate cursor remained unresolved when the ephemeral execution path disappeared. Preserve the journal; a new locked campaign is required for current-source paired evidence.

Evidence: `outputs/autonomy-integration-20260921/science-lab-r24-e9a6de-prereg/preregistration.json`; campaign events at `outputs/autonomy-integration-20260921/science-lab-r24-e9a6de-prereg/campaigns/science-lab-pr1785-e9a6de-r24-20260925/events.jsonl`; supervisor log at `outputs/autonomy-integration-20260921/science-lab-r24-e9a6de-prereg/campaigns/loops/science-lab-pr1785-e9a6de-r24-20260925/supervisor.jsonl`; control scoreboard, AgentEvals, and AgentV bundle under the control run directory. Candidate checkpoint and `train_summary.json` remain under the candidate run directory.

Control result `version_stamp/v1` binds code commit `e9a6de0c6c52e4326252e2c6c76d1abad423d690`, `code_dirty=false`, `harness.model_build.eval=v107`, and `evals.meaningful_program=2.14.0`. The paired endpoint remains unmeasured.
