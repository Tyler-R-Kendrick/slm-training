# Published training corpora

Generated datasets are not sufficient when they remain only under gitignored `outputs/`. Before a training run is used for iteration, publish its version with:

```bash
python scripts/publish_train_data.py --version remediated
# Equivalent management command:
slm-data publish train remediated
```

All model inputs use typed local roots under `outputs/data/<kind>/<dataset-id>/`.
Kinds are `train`, `eval`, `preference`, `annotation`, `trajectory`,
`programspec`, and `mixture`. `slm-data list`, `resolve`, and `verify` provide one
access path across local, Git-published, and legacy data.

The publisher validates and copies the complete immutable snapshot into
`src/slm_training/resources/data/train/<version>/`. It refuses overwrites,
symlinks, unsafe paths, mutable annotations, and individual files of 50 MiB or
more. It does not commit or push; publication finishes through the normal Git
review flow. Training can consume the exact published version with:

```bash
python scripts/train_model.py --train-version remediated --test-dir outputs/data/eval/remediated ...
```

The train summary retains the corpus manifest hash and creating W3C trace ID.
Local `outputs/data/train/<version>` remains preferred when it matches a
published copy. A same-name fingerprint mismatch fails instead of silently
shadowing Git data.

Legacy roots remain readable for one migration window. Preview their safe,
collision-free moves with `slm-data migrate`; apply them explicitly with
`slm-data migrate --apply`. The command never deletes or overwrites a destination.
