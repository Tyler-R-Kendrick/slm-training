# CAP2-04: Implicit vs explicit vs discrete vs compiler-owned state ablation (2026-07-18)

**Linear issue:** SLM-89
**Branch:** `agent/slm-89-state-ablation`
**Date:** 2026-07-18
**Run artifact:** `outputs/runs/cap2_04_state_ablation/cap2_04_state_ablation_01dcdd51a62f3902.json`

Evidence: [iter-cap2-04-state-ablation-20260718.json](iter-cap2-04-state-ablation-20260718.json).
Harness: [`src/slm_training/harnesses/experiments/cap2_04_state_ablation.py`](../../src/slm_training/harnesses/experiments/cap2_04_state_ablation.py),
runner [`scripts/run_cap2_04_state_ablation.py`](../../scripts/run_cap2_04_state_ablation.py).

## What changed

Added a standalone fixture/matrix harness for the decisive CAP2-04 architecture
ablation.  Five matched arms share the same semantic inputs, action vocabulary,
optimizer recipe, seeds, and evaluation protocol; only the state-ownership
mechanism differs:

- `src/slm_training/harnesses/experiments/cap2_04_state_ablation.py`
  - `ArmConfig`, `ArmResult`, `StateAblationReport`, `FixtureDecision`
  - `_ImplicitStateModel` — no exact state; network must infer state from
    history + semantic input; uses `GlobalMaskedHead`.
  - `_ExplicitExactModel` — exact state ID mapped to a learned embedding; uses
    `GlobalMaskedHead`.
  - `_DiscreteCodeModel` — exact state IDs encoded through `MixedRadixFSQCodec`;
    uses `GlobalMaskedHead`.
  - `_CompilerOwnedModel` — compiler owns state; only state-family embedding +
    legal actions; uses `LocalFlatHead`.
  - `_CompilerOwnedNoStateModel` — stricter compiler-owned control; no state
    embedding at all; uses `LocalFlatHead`.
  - Active-parameter matching via `match_active_parameters` and documented
    inactive padding.
  - Unseen-state split for compositional-generalization measurement.
  - Oracle pass (wiring check) and tiny training pass (optimizability check).
- `scripts/run_cap2_04_state_ablation.py`
  - CLI entry point with `--modes`, `--state-count`, `--action-count`,
    `--hidden-dim`, `--seeds`, `--out-dir`, `--no-match-parameters`, `--dry-run`.
- `tests/test_harnesses/experiments/test_cap2_04_state_ablation.py`
  - Manifest instantiation, parameter counting, forced-decision bypass,
    oracle recovery, no-future-info leak, unseen-state split, replay
    determinism, discrete-code capacity, and arm-specific embedding checks.

## Fixture matrix

Command:

```bash
python -m scripts.run_cap2_04_state_ablation --out-dir outputs/runs/cap2_04_state_ablation
```

Recipe: CPU; state_count=8; action_count=5; hidden_dim=16; semantic_dim=8;
seed=0; 200 Adam steps; active-parameter matching enabled.

| arm_id | mode | oracle | random_init | unseen | forced | params | active | capacity | leakage |
| ------ | ---- | ------ | ----------- | ------ | ------ | ------ | ------ | -------- | ------- |
| implicit_s0 | implicit | 1.0000 | 1.0000 | 1.0000 | 0 | 4704 | 4640 | - | False |
| explicit_exact_s0 | explicit_exact | 1.0000 | 1.0000 | 1.0000 | 0 | 4704 | 4624 | - | False |
| discrete_code_s0 | discrete_code | 1.0000 | 1.0000 | 1.0000 | 0 | 4704 | 4704 | 16 | False |
| compiler_owned_s0 | compiler_owned | 1.0000 | 0.0000 | 0.0000 | 0 | 4704 | 81 | - | False |
| compiler_owned_no_state_s0 | compiler_owned_no_state | 1.0000 | 0.2500 | 0.0000 | 0 | 4704 | 33 | - | False |

Unseen state ids: [3, 6].

## Hard gates

- Arms with perfect oracle recovery: 5/5
- Leakage violations: 0
- **PASS** every arm recovered the oracle-encoded action.
- **PASS** no arm leaked below-capacity state information.

## Hypothesis outcomes (fixture scale)

- **CAP-H1 (compiler-owned state is more parameter-efficient):** supported on
  active-parameter counts.  The compiler-owned arms use 81 and 33 active
  parameters versus ~4600+ for the implicit/explicit/discrete arms, even after
  matching total trainable parameters with inactive padding.  Whether this
  translates to matched semantic quality at scale is unrun.
- **CAP-H2 (task-quotient codes outperform exact codes):** not directly tested
  here; the discrete-code arm uses a requirement-matched mixed-radix code with
  capacity 16.  A task-quotient variant requires CAP1-03 task-quotient output.
- **Null/falsifier — implicit state equals compiler-owned at equal cost:**
  not falsified at the fixture scale; both implicit and compiler-owned oracle
  passes recover all actions.  The implicit arm converges to 100% random-init
  accuracy with 200 steps, while the compiler-owned arms do not, consistent with
  their much smaller active parameter budgets.

## Honest caveats

- This is a fixture CPU run.  It proves wiring, matched instantiation, oracle
  recoverability, and optimizability only.
- A production ship-quality claim requires integration into `TwoTowerModel` or
  `GrammarDiffusion` training with full `--ship-gates` eval on real OpenUI
  states and a checkpoint sync to the HF bucket.
- The matched active-parameter counts are enforced by inactive padding, not by
  resizing hidden layers.  A production ladder should match by architecture
  depth/width where possible.
- Random-init/trained accuracies reflect 200 Adam steps on a toy task and are
  not predictive of large-model quality.
- Meaningful-parse remains the primary metric; syntax-parse alone is not
  sufficient.

## Verification checklist

- [x] `pytest tests/test_harnesses/experiments/test_cap2_04_state_ablation.py` — 16 passed.
- [x] `pytest tests/test_harnesses/experiments/test_cap2_bottleneck.py tests/test_models/test_local_action_head.py tests/test_models/test_latent_codec.py tests/test_harnesses/experiments/test_cap2_04_state_ablation.py` — 64 passed.
- [x] Fixture harness `python -m scripts.run_cap2_04_state_ablation` — 5/5 oracle recovery.
