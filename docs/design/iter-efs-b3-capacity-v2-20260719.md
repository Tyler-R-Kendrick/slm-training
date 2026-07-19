# SLM-124 — B3 surface-vs-choice capacity ladder v2 (wiring)

## What
Preregistered, corrected B3 capacity-ladder v2 manifest for EFS3-03. The ladder
reruns the surface-token vs choice-sequence comparison after the E288
choice-native decoder fix, using the EFS1 exposure decision and the
binding-aware meaningful v2 metric as the primary quality axis.

## Matrix registration
- `matrix_set`: `b3-capacity-v2`
- `matrix_version`: `efs3-03-v1`
- `b3_id`: `efs-b3-capacity-v2`

## Arms

| Arm | Representation | Widths | Seeds | Decode fingerprint |
| --- | --- | --- | --- | --- |
| surface | `lexer` | 64, 128, 192 | 0, 1, 2 | `surface_lexer_v1:grammar_constrained=True,ltr_primary=False` |
| choice | `choice` | 64, 128, 192 | 0, 1, 2 | `choice_native_v1:grammar_constrained=True,e288_forced_singleton=True` |

Row set = **2 representations × 3 widths × 3 seeds = 18 primary rows**.

## Frozen base recipe
The base recipe is the E228 legal-candidate-margin recipe extended with the B3
capacity-ladder decode defaults (`mask_pattern=diffusion`,
`grammar_ltr_primary=False`, `context_backend=scratch`). Its SHA-256 is stored in
the manifest.

## Files added
- `src/slm_training/harnesses/experiments/b3_capacity_v2.py`
- `scripts/run_b3_capacity_v2.py`
- `tests/test_harnesses/experiments/test_b3_capacity_v2.py`
- `tests/test_scripts/test_b3_capacity_v2.py`
- `docs/design/iter-efs-b3-capacity-v2-20260719.md`
- `docs/design/iter-efs-b3-capacity-v2-20260719.json`

## Commands

```bash
# Plan only (CPU, no model load)
python -m scripts.run_b3_capacity_v2 --mode plan-only \
  --output-dir outputs/runs/slm124_b3_capacity_v2

# Fixture wiring check
python -m scripts.run_b3_capacity_v2 --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm124_b3_capacity_fixture
```

## Verification
- `pytest tests/test_harnesses/experiments/test_b3_capacity_v2.py -q` → 7 passed
- `pytest tests/test_scripts/test_b3_capacity_v2.py -q` → 4 passed
- `python -m scripts.verify_version_stamps --check` → ok

## Honest caveats
This is **wiring evidence only**. The actual 18-run B3 v2 ladder requires GPU
hosts, durable HF bucket sync per SLM-103, and the EFS1 exposure decision from
SLM-109. The `frontier` mode emits a fixture plan and a clear stderr notice. No
capacity-quality claim or ship gate is made from this artifact.
