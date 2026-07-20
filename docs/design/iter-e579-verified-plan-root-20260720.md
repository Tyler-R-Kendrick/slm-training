# E579 — verifier-gated planned root closure

Date: 2026-07-20  
Status: structural gain; not promotable or ship

E579 replaces the single Stack/EOS preference from E578 with a soft target for
the complete planned root sequence: `Stack([&plan-compatible...], "column")`.
Before any target token receives score, the full prospective choice stream is
decoded and validated. Invalid plans fail closed. Candidate legality is
unchanged, the mechanism remains default-off, and it consumes only prompt-
derived component families plus the visible slot contract.

## Matched result

All arms use clean commit `420b758a`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, choice-codec constrained LTR, 8 generation
steps, 4 attempts, and a 160-token canvas. Every process completed under the
170-second hard cap. Stamps carry eval v16 and TwoTower v16.

| Root weight | Run | meaning-v1 / v2 | structure | recall | reward | AST node / edge | root applications / changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e579-e569-verified-root0-r2` | 0.25 / 0.00 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 0 / 0 | 0/1 |
| 1 | `e579-e569-verified-root1-r2` | 0.25 / 0.00 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 25 / 3 | 0/1 |
| 2 | `e579-e569-verified-root2-r2` | 0.25 / 0.00 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 21 / 5 | 0/1 |
| 4 | `e579-e569-verified-root4-r2` | 0.25 / 0.00 | 0.3013 | 0.3958 | 0.7480 | 0.3976 / 0.2000 | 19 / 8 | 0/1 |

Weight 4 is a real positive structural result: AST-edge F1 rises by 0.20,
AST-node F1 by 0.1143, structural similarity by 0.1325, component recall by
0.125, and reward by 0.0135. Placeholder fidelity and validity remain 0.425
and 0.655. Strict meaning-v2 remains zero and AgentV still fails, so this is
not evidence of semantic readiness and does not authorize promotion.

## Exact invocation

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
  --semantic-role-decode-weight 4 --semantic-plan-decode-weight 1 \
  --semantic-plan-binding-decode-weight 1 \
  --semantic-plan-root-decode-weight <0|1|2|4> \
  --root-reference-arity-decode-weight 1 \
  --root-reference-identity-decode-weight 1
```

## Verdict

Retain the verifier-gated closure scorer default-off as a positive structural
diagnostic. Do not promote or sync a checkpoint. The next experiment should
isolate honest prompt-plan cardinality and schema-value role compatibility:
topology is no longer wholly absent, but strict semantic binding is.

Machine-readable evidence:
[iter-e579-verified-plan-root-20260720.json](iter-e579-verified-plan-root-20260720.json).
