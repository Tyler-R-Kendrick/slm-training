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
slm eval model --test-dir outputs/data/eval/v1 \
  --model twotower --run-id twotower_v1 --ship-gates

# Loss suites (base + OOD)
slm eval loss-suites --checkpoint outputs/runs/<id>/last.pt \
  --test-dir outputs/data/eval/v1

# AgentEvals task cases → scoreboard JSON
slm eval tasks --cases <cases.yaml> --out <scoreboard.json>

# Decode diagnostics
slm eval diagnose --test-dir outputs/data/eval/v1 --out <dir>
```

(`slm eval <action>` ≡ `python -m scripts.evaluate_model` /
`scripts.evaluate_loss_suites` / `scripts.evaluate_tasks` /
`scripts.diagnose_eval`.)

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

- Every evaluation emits AgentEvals JSONL + an AgentV result bundle; no
  alternate run envelope (`docs/design/agentv-evaluation.md`).
- Readiness = `--ship-gates` on the full scoreboard; anything less is wiring
  (`honest-ship-eval` owns readiness language).

## Close out

- Shared duties: [contracts.md](contracts.md).
- Checks: `pytest -q tests/test_harnesses/model_build tests/test_evals`.
