# E577 — preserve plan binding after identity ranking

Date: 2026-07-20  
Status: mechanical success, quality null; not promotable or ship

E576 showed five compatible plan-binding applications but zero immediate
choice changes because learned root-reference identity ranking ran afterward
and permuted the reference score group. E577 composes the same factors in the
opposite order: learned arity, learned identity, then predicted plan binding.
The plan factor remains soft, legal-candidate-preserving, honest, and
default-off.

## Matched result

Both arms use commit `fe706987`, E569 checkpoint SHA
`8254fcf7…c6535f73`, CPU, frozen local HF context, OOD `n=4`, honest visible
slot and semantic-role context, prompt-plan component weight 1, choice-codec
constrained LTR, 8 generation steps, 4 attempts, and a 160-token canvas. Each
process completed under the 170-second hard cap. Clean stamps carry eval v15
and TwoTower v14.

| Binding weight | Run | meaning-v1 / v2 | fidelity | validity | structure | recall | reward | AST node / edge | binding applications / changes | AgentV |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `e577-e569-plan-binding-order-control-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | 0 / 0 | 0/1 |
| 1 | `e577-e569-plan-binding-order1-r1` | 0.25 / 0.00 | 0.4250 | 0.6550 | 0.1688 | 0.2708 | 0.7345 | 0.2833 / 0.0000 | **4 / 2** | 0/1 |

The composition fix succeeds mechanically: binding evidence now changes two
latent reference choices, whereas E576 changed none. However, all four final
programs are byte-for-byte identical to control and every quality metric is an
exact null. The sequential latency samples are retained in JSON but are not a
performance claim.

## Exact invocation

The treatment used:

```bash
timeout --signal=INT --kill-after=10s 170s env \
  SLM_MAX_WALL_MINUTES=3 PYTHONPATH=src \
  OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/src/apps/openui_bridge/cli.mjs \
  AGENTV_RUNNER=/home/codex/repos/slm-training/scripts/run_agentv_eval.mjs \
  /home/codex/repos/slm-training/.venv/bin/python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/remediated --suite ood \
  --run-root outputs/runs --run-id e577-e569-plan-binding-order1-r1 \
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
  --root-reference-arity-decode-weight 1 \
  --root-reference-identity-decode-weight 1
```

The control changes only `--semantic-plan-binding-decode-weight 0` and its run
id.

## Verdict

Retain the corrected factor ordering because it prevents learned identity from
silently erasing the predicted plan signal, but keep the binding factor
default-off. Do not promote or sync a checkpoint. Since two latent changes
still collapse to identical final programs, the next intervention must target
root construction or explicit topology cardinality rather than reference
ranking.

Machine-readable evidence:
[iter-e577-plan-binding-order-20260720.json](iter-e577-plan-binding-order-20260720.json).
