# RICO fixtures

Local slices of the [Rico](http://interactionmining.org/rico) mobile UI dataset
(`shunk031/Rico` on Hugging Face), plus an optional HF stream cache for large
test-harness expansions.

| File | Role |
| --- | --- |
| `semantic_train.jsonl` | Small train slice |
| `semantic_test.jsonl` | Small test slice |
| `semantic_validation.jsonl` | Small validation slice |
| `hf_test_cache.jsonl` | Cached HF `test` screens for offline/large rebuilds |

Rebuild a large disjoint eval set (1500 additional RICO samples):

```bash
python -m scripts.build_test_data \
  --source both \
  --version v1 \
  --train-manifest outputs/train_data/v1/manifest.json \
  --rico-hf-split test \
  --rico-limit 2600 \
  --target-records 1500 \
  --rico-hf-cache fixtures/rico/hf_test_cache.jsonl
```
(Hugging Face: [`shunk031/Rico`](https://huggingface.co/datasets/shunk031/Rico),
config `ui-screenshots-and-hierarchies-with-semantic-annotations`).

| File | HF split | Screens |
| --- | --- | --- |
| `semantic_train.jsonl` | train | 80 |
| `semantic_validation.jsonl` | validation | 20 |
| `semantic_test.jsonl` | test | 40 |

Each line is a screen with mappable semantic elements (`Text`, `Card`, `Text Button`, …)
already exploded from the columnar HF layout. Screenshots are omitted.

Train and test partitions come from **disjoint official HF splits**, so eval screens
never appear in the training slice. The test-data harness additionally rejects any
record whose id / prompt / OpenUI fingerprint appears in the train manifest.

Refresh / expand via:

```bash
python -m scripts.build_train_data --source rico --rico-hf-split train --rico-limit 500
python -m scripts.build_test_data --source rico --rico-hf-split test --rico-limit 100 \
  --train-manifest outputs/train_data/v0/manifest.json
```
