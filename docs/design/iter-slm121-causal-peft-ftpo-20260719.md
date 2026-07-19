# SLM-121 — Causal PEFT FTPO (wiring)

## What
Preregistered causal-adapter FTPO manifest for LDI1-02. The experiment tests
whether updating small PEFT adapters on exact-state causal decision events with
FTPO objectives improves binding-aware meaningful-program rate while preserving
base-model legality and keeping the adapter removable.

## Matrix registration
- `matrix_set`: `causal-peft-ftpo`
- `matrix_version`: `ldi1-02-v1`
- `ftpo_id`: `causal-peft-ftpo`

## Arms

| Objective | Adapter method | Seeds | Purpose |
| --- | --- | --- | --- |
| `unlikelihood` | `lora` | 0, 1, 2 | Negative control over bad legal actions |
| `ftpo_single` | `lora` | 0, 1, 2 | Exactly one good vs one bad action |
| `ftpo_set` | `lora` | 0, 1, 2 | Weighted good × bad margins |
| `legal_set_mass` | `lora` | 0, 1, 2 | Shift legal-space mass from bad set to good set |

The fixture uses LoRA only. DoRA, PiSSA, and AdaLoRA are config-ready but
require PEFT support on the GPU host.

## Frozen base recipe
The base recipe is the E228 legal-candidate-margin recipe
(`iter-e228-candidate-margin-alignment-20260716.json`) extended with adapter and
FTPO hyperparameters. Its SHA-256 is stored in the manifest:
`827379d0c3a9542c5c13acd508ce06669a812a68ccb47b0250423d52b1065575`.

## Files added
- `src/slm_training/harnesses/experiments/causal_peft_ftpo.py`
- `scripts/run_causal_peft_ftpo.py`
- `tests/test_harnesses/experiments/test_causal_peft_ftpo.py`
- `tests/test_scripts/test_causal_peft_ftpo.py`
- `docs/design/iter-slm121-causal-peft-ftpo-20260719.md`
- `docs/design/iter-slm121-causal-peft-ftpo-20260719.json`

## Commands

```bash
# Plan only (CPU, no model load)
python -m scripts.run_causal_peft_ftpo --mode plan-only \
  --output-dir outputs/runs/slm121_causal_peft_ftpo

# Fixture wiring check
python -m scripts.run_causal_peft_ftpo --mode fixture \
  --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
  --output-dir outputs/runs/slm121_causal_peft_fixture
```

## Verification
- `pytest tests/test_harnesses/experiments/test_causal_peft_ftpo.py -q` → 7 passed
- `pytest tests/test_scripts/test_causal_peft_ftpo.py -q` → 4 passed
- `python -m scripts.verify_version_stamps --check` → ok

## Honest caveats
This is **wiring evidence only**. The actual FTPO arm trains require a GPU host,
a causal base checkpoint, an admitted DecisionEventV2 corpus, and durable HF
bucket sync per SLM-103. The `frontier` mode emits a fixture plan and a clear
stderr notice. No adapter quality claim or ship gate is made from this artifact.
