# Published training corpora

Generated datasets are not sufficient when they remain only under gitignored `outputs/`.

`scripts/build_train_data.py` now publishes the built version into
`src/slm_training/resources/data/train/<version>/` **by default** after every
successful build (`--no-publish` opts out for ad-hoc/scratch roots). An
identical rebuild republishes as a no-op; rebuilding the same version with
different content fails loudly — bump `--version` instead of overwriting
evidence. Standalone publication remains available:

```bash
python scripts/publish_train_data.py --version remediated
# Equivalent management command:
slm-data publish train remediated
```

All model inputs use typed local roots under `outputs/data/<kind>/<dataset-id>/`.
Kinds are `train`, `eval`, `preference`, `annotation`, `trajectory`,
`programspec`, `mixture`, and `solver_supervision`. `slm-data list`, `resolve`, and
`verify` provide one access path across local, Git-published, and legacy data.

`solver_supervision` (VSS3-01) is local-only by default: it stores replay-verified
support-set and candidate-cost rows produced by
`scripts/build_solver_supervision.py`. It is excluded from automatic Git publication
because the corpus is large, experiment-specific, and meant to be regenerated from
immutable solver traces rather than snapshotted as a reusable artifact.

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

Solver supervision corpora follow the same layout but are written to
`outputs/data/solver_supervision/<version>/`:

```bash
python scripts/build_solver_supervision.py \
  --trace-root outputs/traces/<run-id> \
  --version v1 \
  --verify-replay
```

Replay verification requires a caller-supplied `ProviderRegistry` when using the
library API; the CLI emits rows from the stored certificates and records whether
replay was requested in the manifest.

Legacy roots remain readable for one migration window. Preview their safe,
collision-free moves with `slm-data migrate`; apply them explicitly with
`slm-data migrate --apply`. The command never deletes or overwrites a destination.
