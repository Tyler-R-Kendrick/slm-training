# SDE0-01: frozen E396/E479 decode-scaffolding × prompt-inventory factorial

**Linear issue:** SLM-161  
**Branch:** `agent/slm-161-decode-scaffolding`  
**Date:** 2026-07-18  
**Run artifact:** `outputs/runs/sde0-01/sde0_01_31e1f4f18ce86ee3.json`

## What changed

Added the SDE0-01 eval-only ablation harness for decomposing learned weights
from decode-time scaffolding.

- `src/slm_training/harnesses/eval/__init__.py`
- `src/slm_training/harnesses/eval/ablate_decode_scaffolding.py`
  - `ScaffoldFactors` — four binary factors: `content_floor`, `prompt_inventory`,
    `semantic_constraints`, `attempts`.
  - `AblateArm` — one factorial cell.
  - `build_stage_a_arms()` — baseline, four one-factor-off arms, and all-off.
  - `resolve_arm_config()` — maps factors onto existing `ModelBuildConfig` /
    `DecodePathSpec` levers and checks compatibility.
  - `_verify_checkpoint()` — fail-closed SHA-256 provenance check.
  - `run_arm()` — fixture mode (config resolution only) or real eval mode via
    `TwoTowerModel.from_checkpoint` + `evaluate_suites`.
  - `run_stage_a()` — runs provenance check, then Stage A, and produces an
    `AblateReport`.
  - `stage_a_needs_stage_b()` — heuristic trigger for the remaining `2^4` cells.
- `scripts/ablate_decode_scaffolding.py`
  - CLI entry point with `--checkpoint`, `--checkpoint-id`, `--checkpoint-sha256`,
    `--checkpoint-remote-uri`, `--output-codec`, `--suites`, `--out-dir`, and
    `--dry-run`.
- `tests/test_harnesses/eval/test_ablate_decode_scaffolding.py`
  - 17 regression tests covering arm generation, factor isolation, config
    resolution, fixture-mode compatibility, Stage B triggering, checkpoint
    SHA-256 provenance, no-inventory contract surfacing, single-attempt mode,
    grammar-only enforcement, and replay determinism.

## Design

The factorial reuses existing infrastructure instead of adding parallel paths:

| Factor | True (E479 control) | False (ablated) |
| --- | --- | --- |
| `content_floor` | `decode_min_content=-1` (auto from inventory) | `decode_min_content=0` |
| `prompt_inventory` | `honest_slot_contract=True`, `slot_contract_in_context=True`, `slot_contract_constrained_decode=True` | all three `False` |
| `semantic_constraints` | `current_exact_or_compiler` decode path, `allow_unconstrained_fallback=False` | `current_native` decode path, `allow_unconstrained_fallback=True` |
| `attempts` | `best_of_n=4`, `generate_max_attempts=3` | `best_of_n=1`, `generate_max_attempts=1` |

The harness never mutates the checkpoint and never feeds `gold.placeholders` to
the model.  When no checkpoint is supplied it runs in fixture mode and verifies
that every arm resolves to a legal, compatible config override set.

## Regression tests

`pytest tests/test_harnesses/eval/test_ablate_decode_scaffolding.py` — 17 passed.

- Stage A produces 6 arms with expected IDs.
- Baseline has all factors enabled; all-off has none.
- Each one-factor-off arm differs from baseline in exactly one factor.
- Baseline resolves to `current_exact_or_compiler` with the E479-equivalent knobs.
- All-off resolves to `current_native` with scaffolding disabled.
- Fixture mode returns compatible for every arm.
- Stage B trigger fires on a non-additive residual and stays off when additivity
  holds.
- Missing or hash-mismatched checkpoint marks every arm incompatible.
- Matching SHA-256 passes the fail-closed provenance check.
- No-inventory arm disables all three slot-contract surfacing fields.
- One-attempt arm sets `best_of_n=1` and `generate_max_attempts=1`.
- All-off arm keeps `grammar_constrained=True`.
- Repeated resolution of the same arm is byte-identical.

## Fixture harness results

Commands:

```bash
python -m scripts.ablate_decode_scaffolding --dry-run
python -m scripts.ablate_decode_scaffolding --out-dir outputs/runs/sde0-01
```

Stage A arms:

| arm | content_floor | prompt_inventory | semantic_constraints | attempts | decode_path | best_of_n | compatible |
| --- | ------------- | ---------------- | -------------------- | -------- | ----------- | --------- | ---------- |
| baseline | True | True | True | True | current_exact_or_compiler | 4 | True |
| one_off_content_floor | False | True | True | True | current_exact_or_compiler | 4 | True |
| one_off_prompt_inventory | True | False | True | True | current_exact_or_compiler | 4 | True |
| one_off_semantic_constraints | True | True | False | True | current_native | 4 | True |
| one_off_attempts | True | True | True | False | current_exact_or_compiler | 1 | True |
| all_off | False | False | False | False | current_native | 1 | True |

Hard gates:

- **PASS** every arm resolved a legal config override set.
- Stage B recommended: `False` (fixture mode uses empty metrics).

## Honest caveats

- This commit delivers the harness and regression tests only.
- The full factorial over the frozen E396/E479 checkpoint has **not** been run
  yet because the checkpoint is not present locally and the suites require GPU
  time beyond the 3-minute hard run cap.
- When the checkpoint is available, run:
  ```bash
  python -m scripts.ablate_decode_scaffolding \
    --checkpoint outputs/runs/e396-balanced-type-head-continuation-r1/checkpoints/last.pt \
    --checkpoint-id e396-balanced-type-head-continuation-r1 \
    --checkpoint-sha256 feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0 \
    --checkpoint-remote-uri hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/ \
    --suites rico_held,rico_dev,bench_small \
    --output-codec choice \
    --out-dir outputs/runs/sde0-01
  ```
- A production claim that the model "learned semantics" requires this ablation
  to report per-metric attribution (`learned`, `scaffolded`, `mixed`, or
  `inconclusive`) against the preregistered gates.

## Verification checklist

- [x] `pytest tests/test_harnesses/eval/test_ablate_decode_scaffolding.py` — 17 passed.
- [x] `.githooks/check-changed` — 17 passed.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `git diff --check` — clean.
- [x] `python -m scripts.ablate_decode_scaffolding --dry-run` — 6 arms described.
- [x] `python -m scripts.ablate_decode_scaffolding --out-dir outputs/runs/sde0-01` — report written.
