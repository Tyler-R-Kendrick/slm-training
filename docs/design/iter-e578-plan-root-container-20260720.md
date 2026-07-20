# E578 — plan-aware Stack root construction

Date: 2026-07-20  
Status: mechanical success, quality null; not promotable or ship

Gold choice streams assemble OOD auth as `Input`, `Input`, `Button`, then
`Stack([&0, &1, &2], "column")`. E578 therefore waits until the generated
top-level elements cover every prompt-plan component family, soft-scores only
legal `Stack` construction, carries plan binding into that Stack's nested child
list, and soft-scores legal EOS after the Stack completes. The mechanism is
honest, candidate-preserving, and default-off.

## Matched result

All arms use commit `2b869c3b`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, prompt-plan component and binding weights 1,
choice-codec constrained LTR, 8 generation steps, 4 attempts, and a 160-token
canvas. Each process completed under the 170-second hard cap. Clean stamps
carry eval v16 and TwoTower v15.

| Root weight | Run | meaning-v1 / v2 | fidelity | validity | structure | recall | reward | AST node / edge | root applications / changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e578-e569-plan-root-control-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 0 / 0 | 0/1 |
| 1 | `e578-e569-plan-root1-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 21 / 1 | 0/1 |
| 2 | `e578-e569-plan-root2-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 14 / 2 | 0/1 |

The intended states activate, and weight 2 changes two latent choices versus
one at weight 1. Nevertheless, every final program is identical to control and
all quality metrics are exact nulls. Sequential latency samples remain in the
JSON and are not a performance claim.

## Exact invocation

Each arm used the E577 recipe plus
`--semantic-plan-root-decode-weight <0|1|2>` and its matching run id, under:

```bash
timeout --signal=INT --kill-after=10s 170s env \
  SLM_MAX_WALL_MINUTES=3 PYTHONPATH=src \
  OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs \
  AGENTV_RUNNER=/home/codex/repos/slm-training/scripts/run_agentv_eval.mjs \
  /home/codex/repos/slm-training/.venv/bin/python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/remediated --suite ood \
  --run-root outputs/runs --run-id <run-id> \
  --checkpoint /tmp/slm-training-e569/outputs/runs/e569-e561-matched-cont48-r1-48s/checkpoints/last.pt \
  --model twotower --device cpu --run-class scratch_matrix --eval-limit 4 \
  --gen-steps 8 --max-attempts 4 --grammar-ltr-primary \
  --grammar-constrained --grammar-ltr-max-tokens 160 --schema-in-context \
  --slot-contract-in-context --semantic-role-contract-in-context \
  --slot-contract-constrained-decode --honest-slot-contract \
  --no-design-md-context --local-files-only \
  --component-plan-decode-weight 0 --slot-component-decode-weight 4 \
  --semantic-role-decode-weight 4 --semantic-plan-decode-weight 1 \
  --semantic-plan-binding-decode-weight 1 \
  --semantic-plan-root-decode-weight <weight> \
  --root-reference-arity-decode-weight 1 \
  --root-reference-identity-decode-weight 1
```

## Verdict

Reject root weights 1 and 2 as quality interventions and keep the bounded
mechanism default-off for diagnostics. Do not promote or sync a checkpoint.
The next topology intervention must use a compiler-validated planned-root
seed/state; further scalar increases are closed.

Machine-readable evidence:
[iter-e578-plan-root-container-20260720.json](iter-e578-plan-root-container-20260720.json).
