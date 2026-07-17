# E316 semantic slot-role synthesis — 2026-07-17

E315 made every declared slot appear in decoded output, but exposed a new
failure: components followed a generic ordinal sequence rather than the slot's
semantic role. A train/eval audit showed why. The E314 corpus maps narrow tails
such as `placeholder` to `Input`, while held requests use ordinary roles such
as `email`, `name`, and `search`.

E316 adds a deterministic `semantic_slots` training-data synthesizer. For
generation records only, it parses the gold AST, associates each declared slot
with its containing component and schema property, and renames the slot from a
small property-role vocabulary selected by stable hash. The mechanism has no
eval IDs, eval layouts, prompt literals, or decoder exceptions. Original rows
remain in the corpus.

## Result

| Measure | E314 | E316 v1 |
| --- | ---: | ---: |
| Rows | 592 | 795 |
| Accepted semantic variants | 0 | 210 |
| Full visible declared contract | 592/592 | 795/795 |
| Verifier rejects | 7 | 13 |
| Build errors | 0 | 0 |
| Eval contamination | 0/19 | 0/19 |
| Train target p95 / max tokens | 93 / 112 | 92 / 112 |
| Targets over 256 tokens | 0 | 0 |
| AgentV diagnostic | 1/1 | 1/1 |

The extra six verifier rejects are derived variants of already rejected or
judge-sensitive rows; they are excluded before publication.

A simple train-derived exact-tail majority probe is deliberately weaker than a
model, but establishes that the data intervention moved the intended signal:

| Suite | E314 coverage / accuracy | E316 coverage / accuracy |
| --- | ---: | ---: |
| smoke | 0.750 / 0.500 | 0.875 / 0.571 |
| held_out | 0.571 / 0.667 | 0.667 / 0.714 |
| adversarial | 0.667 / 1.000 | 0.667 / 1.000 |
| ood | 0.588 / 0.400 | 0.882 / 0.600 |
| rico_held | 1.000 / 1.000 | 1.000 / 1.000 |

All five suites retain component occurrence and type coverage 1.0. The
immutable build reproduced at content fingerprint
`2d01f590b5fef819f3ca3898abb081deb4421e43ed31b0d46e4cd88ccb38a0a2`.

## Recipe

```bash
python -m scripts.build_train_data \
  --source existing \
  --derive-from src/slm_training/resources/data/train/e314_visible_slot_contract_v2/records.jsonl \
  --version e316_semantic_slots_v1 \
  --synthesizer semantic_slots \
  --no-language-contract --no-frontier-artifacts \
  --no-edit-derivatives --repairs-per-program 0 \
  --allow-missing-design-md --prompt-slot-contract --immutable

python -m scripts.diagnose_eval \
  --train-dir outputs/data/train/e316_semantic_slots_v1 \
  --test-dir src/slm_training/resources/data/eval/remediated \
  --ltr-max-tokens 256 \
  --out outputs/runs/e316-semantic-slots-data-r1/diagnostic.json
```

Published snapshot:
`src/slm_training/resources/data/train/e316_semantic_slots_v1`.

**Verdict:** accept v1 for a matched scratch train. This is deterministic
data-build evidence, not checkpoint quality or a ship claim.
