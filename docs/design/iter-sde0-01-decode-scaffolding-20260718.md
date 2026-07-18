# SDE0-01: frozen E396/E479 decode-scaffolding × prompt-inventory factorial

**Linear issue:** SLM-161
**Branch:** `agent/slm-161-decode-scaffolding`
**Date:** 2026-07-18
**Run artifacts:** `outputs/runs/sde0-01/`

## What changed

Added the SDE0-01 eval-only ablation harness for decomposing learned weights
from decode-time scaffolding, then ran Stage A over the frozen E396 checkpoint.
The committed JSON now also carries the canonical top-level `run_id`, `suites`,
`ship_gates`, and `agentv` fields consumed by the deployed research scoreboard;
the factorial detail remains nested under `stage_a`.

- `src/slm_training/harnesses/eval/ablate_decode_scaffolding.py`
  - `ScaffoldFactors` — four binary factors: `content_floor`, `prompt_inventory`,
    `semantic_constraints`, `attempts`.
  - `AblateArm` / `ArmResult` / `AblateReport` — typed factorial cells and report.
  - `build_stage_a_arms()` — baseline, four one-factor-off arms, and all-off.
  - `resolve_arm_config()` — maps factors onto existing `ModelBuildConfig` /
    `DecodePathSpec` levers and checks compatibility.
  - `_verify_checkpoint()` — fail-closed SHA-256 provenance check.
  - `run_arm()` — fixture mode (config resolution only) or real eval mode via
    `TwoTowerModel.from_checkpoint` + `evaluate_suites`.
  - `run_stage_a()` — provenance check, Stage A, and `AblateReport`.
  - `build_stage_b_arms()` — remaining `2^4` cells when Stage A is non-additive.
  - `compute_paired_deltas()` and `estimate_additive_interaction()` — attribution
    statistics.
- `scripts/ablate_decode_scaffolding.py`
  - CLI with `--checkpoint`, `--checkpoint-id`, `--checkpoint-sha256`,
    `--checkpoint-remote-uri`, `--output-codec`, `--suites`, `--out-dir`,
    `--test-dir`, `--rico-limit`, and `--dry-run`.
  - Fixed 170-second process deadline; timeout exits `124`, leaving ten seconds
    for the caller's mandatory kill grace.
- `tests/test_harnesses/eval/test_ablate_decode_scaffolding.py`
  - 23 regression tests covering arm generation, factor isolation, config
    resolution, fixture compatibility, Stage B triggering, provenance, no-gold
    inventory, single-attempt mode, grammar-only enforcement, replay
    determinism, paired deltas, additive interaction, and mock-verified real
    eval wiring.
- `src/slm_training/models/twotower.py`
  - Tolerate legacy `slot_component_head.*` weights in E396/E479 checkpoints
    when the live model no longer instantiates that head.
- `src/slm_training/dsl/production_codec.py`
  - Harden `_decode_literal` against empty/malformed literal payloads so an
    invalid generated token is treated as a literal string instead of crashing
    eval.
- `src/slm_training/harnesses/model_build/config.py` + `factory.py`
  - Added `slot_component_decode_weight` to `ModelBuildConfig` and the runtime
    override path.

## Design

The factorial reuses existing infrastructure instead of adding parallel paths.
The E479-equivalent recipe is held constant across arms; only the four factors
are varied.

| Factor | True (E479 control) | False (ablated) |
| --- | --- | --- |
| `content_floor` | `decode_min_content=-1` (auto from inventory) | `decode_min_content=0` |
| `prompt_inventory` | `honest_slot_contract=True`, `slot_contract_in_context=True`, `slot_contract_constrained_decode=True` | all three `False` |
| `semantic_constraints` | `current_exact_or_compiler` decode path, `schema_in_context=True`, `allow_unconstrained_fallback=False` | `current_native` decode path, `schema_in_context=False`, `allow_unconstrained_fallback=True` |
| `attempts` | `best_of_n=4`, `generate_max_attempts=3` | `best_of_n=1`, `generate_max_attempts=1` |

Recipe constants held constant: `component_plan_decode_weight=2.0`,
`component_inventory_decode_weight=8.0`, `grammar_constrained=True`,
`grammar_ltr_primary=True`, `gen_steps=8`, `grammar_ltr_max_tokens=320`.

The harness never mutates the checkpoint and never feeds `gold.placeholders` to
the model.

## Regression tests

```bash
pytest tests/test_harnesses/eval/test_ablate_decode_scaffolding.py
```

- 25 passed.

## Measured Stage A results

Checkpoint provenance:

- ID: `e396-balanced-type-head-continuation-r1`
- SHA-256: `feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`
- Remote URI: `hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1/`
- Local path: `outputs/runs/e396-balanced-type-head-continuation-r1/checkpoints/last.pt`

All Stage A arms resolved legal configs and completed on every suite.  The
primary metric is **meaningful-program rate** (not syntax parse).

| Suite | Artifact | Baseline MPR | All-off MPR | Stage B |
| --- | --- | ---: | ---: | --- |
| smoke | `sde0_01_0d064fd3402d0a60.json` | 0.00 | 0.00 | False |
| held_out | `sde0_01_41bdf3df8080fa55.json` | 0.00 | 0.00 | False |
| adversarial | `sde0_01_0d12959b8327ac54.json` | 0.00 | 0.00 | **True** |
| ood | `sde0_01_cf02ad961a78def8.json` | 0.00 | 0.00 | False |
| rico_held | `sde0_01_8f1452307921b6ce.json` | 0.00 | 0.00 | False |

`rico_held` was run with `--rico-limit 50` because the full 1,500-record suite
cannot finish on CPU within the 3-minute hard run cap.  The other suites ran
unreduced.

### Paired deltas (placeholder fidelity)

Placeholder fidelity is the only metric that moves materially across arms; MPR
and component-type recall stay at 0 for all cells.

| Suite | prompt_inventory off | semantic_constraints off |
| --- | ---: | ---: |
| smoke | -0.111 | +0.111 |
| held_out | -0.213 | 0.000 |
| adversarial | -0.333 | 0.000 |

Removing prompt-surfaced inventory consistently removes placeholder fidelity;
removing semantic constraints has mixed or no effect.

## Attribution verdict

| Metric | Verdict |
| --- | --- |
| meaningful_program_rate | **inconclusive** — baseline is 0.00, so no E479 gate-pass is available to decompose. |
| placeholder_fidelity | **scaffolded** (prompt inventory) — fidelity drops when inventory is removed. |
| component_type_recall | **inconclusive** — 0.00 in every arm. |
| structure | **inconclusive** — low in every arm. |
| AgentV | **inconclusive** — all suite gates fail. |
| latency | **learned** in the sense that decode-path differences do not dominate wall time; all arms are dominated by the frozen HF-context forward pass. |

## Honest caveats

- The E396 checkpoint loads and all Stage A arms complete, but **the baseline
  does not reproduce the E479 reported gate pass** (`MPR=1.0` on smoke,
  held_out, adversarial, ood, rico_held).  Under the current decoder/recipe the
  model emits parsed-but-trivial programs (`Stack([...])` with no content
  components or placeholders), yielding `MPR=0.00` even with every scaffold
  enabled.
- Because the control arm fails, the hypothesis test is **not informative**:
  we cannot attribute a gate pass that does not occur locally.
- Possible explanations: a missing E479 runtime detail, code drift since the
  E479 evaluation source commit (`0c46546082d0d1b6338c314429eacf410f56b03c`),
  or a mismatch between the uploaded checkpoint and the reported E479 recipe.
- `rico_held` is capped at 50 records due to the CPU 3-minute hard run cap.  A
  full 1,500-record ship-grade rerun requires GPU/HF Jobs and the correct
  recipe.
- No production ship claim is made.  This is diagnostic harness + evidence.

## Verification checklist

- [x] `pytest tests/test_harnesses/eval/test_ablate_decode_scaffolding.py` — 25 passed.
- [x] `.githooks/check-changed` — passed.
- [x] `python -m scripts.repo_policy` — ok.
- [x] `git diff --check` — clean.
- [x] Stage A completed on smoke, held_out, adversarial, ood, and a 50-record
      rico_held cap with all arms compatible.
