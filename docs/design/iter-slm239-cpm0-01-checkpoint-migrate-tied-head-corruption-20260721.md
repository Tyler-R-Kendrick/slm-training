# SLM-239 (CPM0-01): checkpoint-migrate output-head corruption probe (slm239-cpm0-01-checkpoint-migrate-tied-head-corruption-20260721)

**Matrix set:** `slm239_checkpoint_migrate_tied_head_corruption`
**Version:** `cpm0-01-v1`
**Status:** fixture
**Claim class:** wiring

## Hypothesis

The shipped, unmodified migrate_twotower_checkpoint correctly remaps every vocab-indexed weight (token embedding AND output head) by shared token string when migrating against a new train-records set that happens to produce the same vocabulary size as the source checkpoint but a different first-occurrence token order.

## Falsifier

For a majority of seeds, either (a) with tie_output_embedding=True the post-migration on-disk denoiser.tok.weight rows for tokens whose id shifted no longer match the token-string-correct source row (the lm_head naive copy clobbers the shared, already-remapped storage), or (b) with tie_output_embedding=False the post-migration on-disk denoiser.lm_head.weight rows for shifted tokens do not match the token-string-correct source row (the untied output head is never remapped at all).

## Honest caveats

- Fixture/wiring evidence only: a tiny untrained scratch-backend TwoTowerModel (d_model=16, 1 layer per tower) with a hand-built ~20-30 token vocabulary. No production checkpoint, real train-records corpus, or GPU run is used.
- The 'same vocab size, different order' precondition is engineered by permuting a fixed record set's order rather than sampled from an organic append/drop history; it demonstrates the mechanism is real and reachable, not how often it fires against actual production data drift.
- Only shared tokens whose id actually shifted between the old and (re-tokenized) new vocabulary are scored; unseen new-only tokens (correctly left randomly initialized) are excluded because there is no single correct value to compare against.
- This harness calls migrate_twotower_checkpoint exactly as shipped -- no line of src/slm_training/models/checkpoint_migrate.py is modified, stubbed, or monkeypatched.

## Sweep

- seeds: [0, 1, 2, 3, 4]
- tie_output_embedding arms: [True, False]
- correct-fraction corruption threshold: 5%

## Per-probe results

| tie | seed | old vocab | new vocab | shifted tokens | tok correct frac | lm_head correct frac | tok==raw old (whole) | corrupted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | 0 | 91 | 91 | 78 | 0.00 | 0.00 | True | True |
| True | 1 | 91 | 91 | 78 | 0.00 | 0.00 | True | True |
| True | 2 | 91 | 91 | 76 | 0.00 | 0.00 | True | True |
| True | 3 | 91 | 91 | 76 | 0.00 | 0.00 | True | True |
| True | 4 | 91 | 91 | 79 | 0.00 | 0.00 | True | True |
| False | 0 | 91 | 91 | 78 | 1.00 | 0.00 | False | True |
| False | 1 | 91 | 91 | 78 | 1.00 | 0.00 | False | True |
| False | 2 | 91 | 91 | 76 | 1.00 | 0.00 | False | True |
| False | 3 | 91 | 91 | 76 | 1.00 | 0.00 | False | True |
| False | 4 | 91 | 91 | 79 | 1.00 | 0.00 | False | True |

## Summary

- tied arm corrupted: 5/5
- untied arm corrupted: 5/5
- any vocab-size-mismatch seed (invalidates precondition): False

## Disposition

**gap_confirmed**

All 5/5 tie_output_embedding=True seeds showed the predicted clobber: denoiser.tok.weight's token-string remap was overwritten wholesale by the raw, un-remapped old matrix via the aliased denoiser.lm_head.weight key. All 5/5 tie_output_embedding=False seeds showed the predicted drift: denoiser.tok.weight was correctly remapped but denoiser.lm_head.weight was never remapped at all, so the output head silently misaligns with the (correct) input embedding. migrate_twotower_checkpoint's per-token remap covers only the '.tok.weight' key; every other vocab-indexed weight, including the output head under both tying modes, is copied wholesale whenever old and new vocab sizes happen to coincide.

## Go / no-go decision

**No-go for trusting migrate_twotower_checkpoint's output head under vocab reorder; genuine gap, not a promotion candidate.** This is wiring/fixture evidence over a tiny untrained scratch model with a hand-built ~25-30 token vocabulary. It exercises the real, unmodified production migration function end to end (construct, save, migrate, reload from disk) rather than re-deriving the claim analytically. A `gap_confirmed` disposition means anyone who runs `scripts/migrate_checkpoint.py` against a train-records set with a coincidentally-matching vocab size gets a checkpoint whose output head is silently misaligned with its (correctly remapped) input embedding -- with tying on, the *input* embedding also reverts to the old, wrong order. This is flagged to the maintainer as a real bug in migrate_twotower_checkpoint's per-key remap coverage, not acted on here: this harness makes no change to src/slm_training/models/checkpoint_migrate.py.

## Reproducibility

```bash
python -m scripts.run_slm239_checkpoint_migrate_tied_head_corruption --mode plan-only
python -m scripts.run_slm239_checkpoint_migrate_tied_head_corruption --mode fixture
```

