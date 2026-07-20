# E582 — distinct repeated-slot instances

Date: 2026-07-20  
Status: partial structural gain; not promotable or ship

E582 softly closes a repeated predicted component after it consumes its first
visible slot while more instances of that family remain. This prevents one
component from absorbing multiple authored roles before the next instance can
be generated. Candidate legality is unchanged and the scorer remains
default-off.

## Matched result

Both arms use clean commit `56e6975f`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, root weight 4, choice-codec constrained LTR,
8 generation steps, 4 attempts, and a 160-token canvas. Each process completed
under the 170-second hard cap. Stamps carry eval v16 and TwoTower v19.

| Plan weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | plan changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e582-e569-distinct-slots0-r1` | 0.00 / 0.00 | 0.3417 / 0.6050 | 0.1250 | 0.1458 | 0.7095 | 0.1833 / 0.0000 | 0 | 0/1 |
| 4 | `e582-e569-distinct-slots4-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.3119 | 0.4583 | 0.7510 | 0.4264 / 0.1667 | 6 | 0/1 |

Versus the clean no-plan control, weight 4 improves meaning-v1 by 0.25,
fidelity by 0.0833, validity by 0.05, structure by 0.1869, component recall by
0.3125, reward by 0.0415, AST-node F1 by 0.2431, and AST-edge F1 by 0.1667.

The targeted auth record reaches AST-node F1 0.75, AST-edge F1 0.6667, and an
exact reference graph. The first Input is now distinct, but the email slot is
assigned to TextContent instead of the still-required second Input:

```openui
root = Stack([v0, v3, v1, ":ood.auth.create"], "column", ":ood.auth.create")
v0 = Input(":ood.auth.name")
v3 = Button(":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create")
v1 = TextContent(":ood.auth.email", ":ood.auth.create")
```

Strict meaning-v2 remains zero because auth still has a placeholder
semantic-role mismatch, schema-value role mismatches, and placeholder spam.

## Reproduction

Each arm used:

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
  --semantic-role-decode-weight 4 --semantic-plan-decode-weight <0|4> \
  --semantic-plan-binding-decode-weight 1 \
  --semantic-plan-root-decode-weight 4 \
  --root-reference-arity-decode-weight 1 \
  --root-reference-identity-decode-weight 1
```

## Verdict

Retain the legality-preserving reservation default-off as a partial structural
diagnostic. Do not promote or sync a checkpoint. The next experiment should
isolate visible-slot-to-predicted-family scoring so the email role selects the
second Input rather than a merely schema-compatible family.

Machine-readable evidence:
[iter-e582-distinct-slot-instances-20260720.json](iter-e582-distinct-slot-instances-20260720.json).
