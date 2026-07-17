# E307 component-prompt synthesis — 2026-07-17

E306 found complete component-type support but weak prompt-to-component
alignment. E307 adds a deterministic `component` synthesis mode that converts
each eligible train target into ordinary prose describing its component
inventory and terminal content concepts. It uses parsed target-independent
counts and generic CamelCase/placeholder normalization; it contains no eval
prompt, component-specific template, or decoder exception.

The accepted immutable corpus derives only from E218 train records. Language
contract regeneration, frontier expansion, edit derivation, and repair
derivation were disabled so the intervention preserves all 480 source rows and
adds only judged prompt variants.

## Result

| Measure | E218 | E307 v4 |
| --- | ---: | ---: |
| Kept rows | 480 | 592 |
| Generation rows | 119 | 231 |
| Generation prompts naming any target component | 15 | 127 |
| Generation prompts naming the full target inventory | 1 | 111 |
| Component-prompt rows | 0 | 112 |
| Build errors | — | 0 |
| Independent-verifier rejects | — | 7 |
| Eval contamination | — | 0/19 |

The seven rejected derivatives failed the existing G11 independent judge and
were not written. Two valid `Buttons` examples are deliberately not counted as
full inventory matches by the ambiguity-aware plural scanner, but pass the
admission judge.

The build reproduced byte-for-byte at content fingerprint
`62f408c85ba7fa67606bd8deecb904cecb58f78255fc4ef101b7e7c4afa60259`.
All five suites retain component occurrence/type coverage 1.0. Train target
p95/max are 93/112 tokens, with zero rows above the 256-token budget. The
canonical diagnostic published AgentEvals plus AgentV evidence at 1/1.

Three controls were rejected before acceptance: v1 applied a redundant global
parent cap and collapsed existing task families; v2 regenerated rather than
preserved the language-contract rows; v3 emitted raw CamelCase rather than
natural component words. Only corrected v4 is committed.

## Recipe

```bash
python -m scripts.build_train_data \
  --source existing \
  --derive-from src/slm_training/resources/data/train/e218_schema_normalized_judge_v5/records.jsonl \
  --version e307_component_prompt_v4 \
  --synthesizer component \
  --no-language-contract --no-frontier-artifacts \
  --no-edit-derivatives --repairs-per-program 0 \
  --allow-missing-design-md --immutable
```

Published snapshot:
`src/slm_training/resources/data/train/e307_component_prompt_v4`.

**Verdict:** accept E307 as leak-free deterministic training data and run a
matched scratch training comparison. No model was evaluated here, so this is
not a ship or checkpoint-quality claim.
