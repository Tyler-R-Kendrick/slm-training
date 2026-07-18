---
name: phase-eval
description: Evaluate checkpoints on honest multi-suite scoreboards — evaluate_model with ship gates, loss suites, task evals, and decode diagnostics — emitting AgentEvals JSONL plus AgentV bundles. Use when measuring a model or preparing a readiness claim.
---

# Evaluation phase

Multi-suite, honesty-constrained evaluation. Owner:
`src/slm_training/harnesses/model_build/` (`eval_runner.py`, `ship_gates.py`).

## Prerequisites

- `outputs/data/eval/<version>/` (test-data phase); a checkpoint/run to score.
- `npm ci` once before Python eval commands (pinned AgentV SDK); eval paths
  write AgentV bundles under `<run-dir>/agentv/` automatically.

## Commands

```bash
# Honest multi-suite scoreboard with ship gates
python -m scripts.evaluate_model --test-dir outputs/data/eval/v1 \
  --model twotower --run-id twotower_v1 --ship-gates

# Loss suites (base + OOD)
python -m scripts.evaluate_loss_suites --checkpoint outputs/runs/<id>/last.pt \
  --test-dir outputs/data/eval/v1

# AgentEvals task cases → scoreboard JSON
python -m scripts.evaluate_tasks --cases <cases.yaml> --out <scoreboard.json>

# Decode diagnostics
python -m scripts.diagnose_eval --test-dir outputs/data/eval/v1 --out <dir>
```

## Key flags

`--suite`, `--eval-limit`, `--device`, `--fail-under-parse-rate`,
`--fail-under-placeholder-fidelity`, `--fail-under-reward-score`,
`--compiler-search-*` decode knobs.

## Outputs

Scoreboards + AgentV bundles beside domain JSON under `outputs/runs/<run-id>/`.
Metrics: meaningful parse, strict `placeholder_fidelity`,
`structural_similarity`, composite `reward_score`; suites smoke / held_out /
rico_held / adversarial / ood.

## Gates & invariants

- Readiness requires `--ship-gates` on the full scoreboard — smoke-only or
  fixture fail-unders are wiring, not ship evidence (`honest-ship-eval`).
- Every evaluation emits AgentEvals JSONL + an AgentV result bundle; no
  alternate run envelope.

## Close out

- Iron law: matching `docs/design/` JSON + markdown scoreboard
  (`documenting-experiment-results`); readiness language → `honest-ship-eval`.
- Checks: `pytest -q tests/test_harnesses/model_build tests/test_evals`.
