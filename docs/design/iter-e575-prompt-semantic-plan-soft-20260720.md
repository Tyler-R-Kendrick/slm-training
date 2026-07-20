# E575 — prompt-derived SemanticPlanV1 soft scoring

Date: 2026-07-20  
Status: completed decode Pareto; not promotable or ship

E575 connects the SLM-146 compiler bridge to the production choice-codec
decoder without a gold channel. Explicit component names are extracted from
the visible prompt using the official schema vocabulary and serialized into a
predicted partial `SemanticPlanV1`. `OpenUISemanticPlanCompiler` converts that
plan into soft features for legal root and bound component actions. The scorer
does not add or remove candidates, and prompts with no recognized component
mentions compile to the unchanged baseline.

## Matched recipe

All three clean r3 arms use commit `11813e60`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot/semantic-role context, choice-codec constrained LTR, 8 generation steps,
4 attempts, a 160-token canvas, and weights 4/4/1/1 for slot component,
semantic role, root-reference arity, and root-reference identity. Every process
was capped at 170 seconds and completed; the result stamps include eval v14 and
TwoTower v12 with `code_dirty=false`.

| Prompt-plan weight | Run | meaning-v1 | meaning-v2 | fidelity | validity | structure | component recall | reward | AST node / edge | applications / changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e575-e569-prompt-plan-control-r3` | 0.00 | 0.00 | 0.3417 | 0.6050 | 0.1250 | 0.1458 | 0.7095 | 0.1833 / 0.0000 | 0 / 0 | 0/1 |
| **1** | `e575-e569-prompt-plan1-r3` | **0.25** | 0.00 | **0.4250** | **0.6550** | **0.1688** | **0.2708** | **0.7345** | **0.2833 / 0.0000** | 52 / 3 | 0/1 |
| 2 | `e575-e569-prompt-plan2-r3` | 0.25 | 0.00 | 0.2583 | 0.5550 | 0.1688 | 0.2708 | 0.6845 | 0.2833 / 0.0000 | 51 / 5 | 0/1 |

Weight 1 changes the auth output from an unrelated `TextArea` root to the
prompt-mentioned `Input`, recovering the only meaningful-v1 pass. Relative to
the matched control, it improves fidelity by 0.0833, validity by 0.05,
structure by 0.0438, component recall by 0.125, reward by 0.025, and AST-node
F1 by 0.10 while syntax remains 1.0. The sequential latency samples are
reported in the JSON but are not treated as a performance claim.

Weight 2 causes five choice interventions instead of three but loses 0.1667
fidelity, 0.10 validity, and 0.05 reward versus weight 1. This closes the small
0/1/2 scalar ladder: stronger prompt-plan bias is not justified.

## Exact invocation

Each arm used the following command with `<weight>` set to 0, 1, or 2 and the
matching r3 run id:

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
  --semantic-role-decode-weight 4 --semantic-plan-decode-weight <weight> \
  --root-reference-arity-decode-weight 1 \
  --root-reference-identity-decode-weight 1
```

## Verdict

Retain the generalized, legality-preserving scorer and weight 1 as a local
decode Pareto, default-off. Do not promote or sync: binding-aware meaning-v2
and AgentV remain zero. The next semantic-plan intervention should predict
topology or binding factors; increasing this component-family scalar is closed.

Machine-readable evidence:
[iter-e575-prompt-semantic-plan-soft-20260720.json](iter-e575-prompt-semantic-plan-soft-20260720.json).
