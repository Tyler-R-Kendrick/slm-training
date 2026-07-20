# E576 — prompt-plan binding soft scoring

Date: 2026-07-20  
Status: completed null; not promotable or ship

E576 tests a generalized binding factor after E575 improved component choice
but left AST-edge F1 and strict meaning-v2 at zero. During the terminal
structural root list, the decoder soft-ranks only legal, unused references
whose already-generated component family matches a role in the prompt-derived
predicted `SemanticPlanV1`. It never adds or removes candidates, does not use
gold topology, is neutral in nested lists, and remains default-off.

## Matched result

All arms use commit `4b60437e`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, component-plan weight 1, choice-codec
constrained LTR, 8 generation steps, 4 attempts, and a 160-token canvas. Each
process completed under the 170-second hard cap. The clean stamps carry eval
v15 and TwoTower v13.

| Binding weight | Run | meaning-v1 / v2 | fidelity | validity | structure | recall | reward | AST node / edge | binding applications / changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e576-e569-plan-binding-control-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 0 / 0 | 0/1 |
| 1 | `e576-e569-plan-binding1-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 5 / 0 | 0/1 |
| 2 | `e576-e569-plan-binding2-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 5 / 0 | 0/1 |

Both treatments are exact quality nulls. Weight 2 changes the subsequent
root-arity and root-identity intervention counts, showing an interaction with
those learned heads, but the binding factor itself never changes an immediate
argmax and all final programs and scored metrics remain unchanged. Sequential
latency samples are retained in the JSON and are not a performance claim.

## Exact invocation

Each arm used this command with `<weight>` set to 0, 1, or 2:

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
  --semantic-plan-binding-decode-weight <weight> \
  --root-reference-arity-decode-weight 1 \
  --root-reference-identity-decode-weight 1
```

## Verdict

Reject weights 1 and 2 as quality interventions; retain the bounded mechanism
default-off for diagnostics only. Do not promote or sync a checkpoint. The
next intervention must change root construction or predict explicit topology
cardinality rather than further increasing reference scores.

Machine-readable evidence:
[iter-e576-prompt-plan-binding-soft-20260720.json](iter-e576-prompt-plan-binding-soft-20260720.json).
