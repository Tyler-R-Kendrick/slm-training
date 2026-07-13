# Playground demo checkpoint

The web playground loads `playground_demo/last.pt` (plus `.tokenizer.json` and `.meta.json` sidecars).

These files are **committed to source control** so a fresh clone can run the annotate UI and Playwright e2e without training first.

Regenerate after changing demo records or model defaults:

```bash
python -m scripts.bootstrap_playground --force
git add fixtures/checkpoints/playground_demo/
git commit -m "Refresh playground demo checkpoint"
```

Full training runs still write checkpoints under `outputs/runs/<run-id>/checkpoints/` (gitignored).
