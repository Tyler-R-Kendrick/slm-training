# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation.

This cycle ships **three harnesses only** (no TwoTower model yet):

1. **Training-data** — build/validate versioned train corpora
2. **Testing-data** — build held-out / adversarial / OOD eval suites
3. **Model-building** — train/eval shell with a stub plug-in

See [docs/design/openui-twotower.md](docs/design/openui-twotower.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start (offline)

```bash
# 1. Build train artifacts from fixtures
python -m scripts.build_train_data --version v0

# 2. Build test suites (leakage check vs train)
python -m scripts.build_test_data --version v0 \
  --train-manifest outputs/train_data/v0/manifest.json

# 3. Train stub model
python -m scripts.train_model \
  --train-dir outputs/train_data/v0 \
  --steps 2

# 4. Evaluate stub on smoke suite
python -m scripts.evaluate_model \
  --test-dir outputs/test_data/v0 \
  --suite smoke \
  --checkpoint outputs/runs/latest/checkpoints/last.pt
```

```bash
pytest
```

## Layout

```
src/slm_training/dsl/           # shared grammar / schema
src/slm_training/harnesses/     # train_data, test_data, model_build
scripts/                        # CLIs
fixtures/                       # seed pairs for offline CI
docs/design/                    # architecture + contracts
```
