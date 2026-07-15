# Published training corpora

Generated datasets are not sufficient when they remain only under gitignored `outputs/`. Before a training run is used for iteration, publish its version with:

```bash
python scripts/publish_train_data.py --version remediated
```

The publisher copies `records.jsonl`, `manifest.json`, and `stats.json` into `src/slm_training/resources/train_data/<version>/`. The dashboard reads those committed versions under Training Data, including searchable records. Training can consume the exact published version with:

```bash
python scripts/train_model.py --train-version remediated --test-dir outputs/test_data/remediated ...
```

The train summary retains the corpus manifest hash, so a run can be tied back to the source-controlled dataset. Local `outputs/train_data/<version>` remains preferred when it exists, allowing exploratory regeneration before an explicit publish.
