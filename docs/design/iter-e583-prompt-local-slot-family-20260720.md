# E583 — prompt-local slot-family scoring

Date: 2026-07-20  
Status: negative; not promotable or ship

E583 extends the honest visible semantic-role candidates beyond exact schema
property names. It associates a slot's terminal role with a prompt-mentioned
component family when both occur within three tokens in the same authored
clause. Appended inventory, component, and semantic-role lines are excluded.
Candidate legality is unchanged and the scorer remains default-off.

## Matched result

Both arms use clean commit `913733df`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, component-plan weight 4, root weight 4,
choice-codec constrained LTR, 8 generation steps, 4 attempts, and a 160-token
canvas. Each process completed under the 170-second hard cap. Stamps carry
eval v16 and TwoTower v20.

| Role weight | Run | meaning-v1 / v2 | fidelity / validity | structure | recall | reward | AST node / edge | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e583-e569-slot-family0-r1` | 0.25 / 0.00 | 0.5083 / 0.7050 | 0.3119 | 0.4583 | 0.7760 | 0.4264 / 0.1667 | 0/1 |
| 4 | `e583-e569-slot-family4-r1` | 0.25 / 0.00 | 0.4250 / 0.6550 | 0.3119 | 0.4583 | 0.7510 | 0.4264 / 0.1667 | 0/1 |

The treatment is quality-null on meaning, structure, recall, and AST metrics,
while regressing fidelity by 0.0833, validity by 0.05, and reward by 0.025.
The auth prediction is identical in both arms:

```openui
root = Stack([v0, v3, v1, ":ood.auth.create"], "column", ":ood.auth.create")
v0 = Input(":ood.auth.name")
v3 = Button(":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create", ":ood.auth.create")
v1 = TextContent(":ood.auth.email", ":ood.auth.create")
```

The only changed program is modal: the treatment replaces the control's
`:ood.modal.body` with `:ood.modal.confirm`, explaining the fidelity and
validity regression. Strict meaning-v2 remains zero and AgentV fails.

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
  --semantic-role-decode-weight <0|4> --semantic-plan-decode-weight 4 \
  --semantic-plan-binding-decode-weight 1 \
  --semantic-plan-root-decode-weight 4 \
  --root-reference-arity-decode-weight 1 \
  --root-reference-identity-decode-weight 1
```

## Verdict

Record the generalized visible association, but reject role weight 4 for this
recipe. Keep it default-off and do not promote or sync a checkpoint. The next
experiment should isolate learned-versus-visible role score composition; a
larger additive scalar is not justified by this result.

Machine-readable evidence:
[iter-e583-prompt-local-slot-family-20260720.json](iter-e583-prompt-local-slot-family-20260720.json).
