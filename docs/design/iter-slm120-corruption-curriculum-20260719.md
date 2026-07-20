# SLM-120 — Near-solved semantic corruption curriculum (wiring)

## What
Preregistered curriculum manifest and `CorruptionTraceV2` schema for EFS3-02.
The experiment tests whether injecting a controlled share of one- and two-error
semantic states improves recovery and fixed-point stability without harming
from-scratch generation.

## Matrix registration
- `matrix_set`: `corruption-curriculum`
- `matrix_version`: `efs3-02-v1`
- `curriculum_id`: `near-solved-semantic`

## Arms

| Arm | Near-solved share (S1+S2) | Purpose |
| --- | --- | --- |
| A   | 0%   | clean control |
| B   | 5%   | low intervention |
| C   | 10%  | medium intervention |
| D   | 15%  | high intervention |
| E   | 30%  | stress / copying-failure test |

Within the near-solved share, S1 and S2 are split 50/50.

## Severity taxonomy

| Level | Name | Description |
| --- | --- | --- |
| S0 | `S0_clean` | no corruption; stability/identity target |
| S1 | `S1_near_solved_1` | exactly one semantic corruption |
| S2 | `S2_near_solved_2` | exactly two semantic corruptions |
| S3 | `S3_medium` | 3–5 semantic corruptions |
| S4 | `S4_heavy` | current full/high-mask corruption |

## Frozen base recipe
The base recipe is the E228 legal-candidate-margin recipe
(`iter-e228-candidate-margin-alignment-20260716.json`). Its SHA-256 is stored in
the manifest.

## Files added
- `src/slm_training/data/corrupt/trace.py`
- `src/slm_training/harnesses/experiments/corruption_curriculum.py`
- `scripts/run_corruption_curriculum.py`
- `tests/test_data/test_corrupt_trace.py`
- `tests/test_harnesses/experiments/test_corruption_curriculum.py`
- `tests/test_scripts/test_corruption_curriculum.py`
- `docs/design/iter-slm120-corruption-curriculum-20260719.md`
- `docs/design/iter-slm120-corruption-curriculum-20260719.json`

## Commands

```bash
# Plan only (CPU, no model load)
python -m scripts.run_corruption_curriculum --mode plan-only \
  --output-dir outputs/runs/slm120_corruption_curriculum

# Fixture wiring check
python -m scripts.run_corruption_curriculum --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm120_corruption_fixture
```

## Verification
- `pytest tests/test_data/test_corrupt_trace.py -q` → 6 passed
- `pytest tests/test_harnesses/experiments/test_corruption_curriculum.py -q` → 9 passed
- `pytest tests/test_scripts/test_corruption_curriculum.py -q` → 3 passed
- `python -m scripts.verify_version_stamps --check` → ok

## Honest caveats
This is **wiring evidence only**. The actual A–D arm trains require a GPU host,
the EFS1-decided base recipe/checkpoint, and durable HF bucket sync per SLM-103.
The `frontier` mode emits a fixture plan and raises a clear stderr message. No
curriculum claim or ship gate is made from this artifact.
