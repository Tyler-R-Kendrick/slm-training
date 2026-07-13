# OpenUI TwoTower — Design Spec

## Problem

Build a small, on-device-friendly specialist that generates **placeholder-augmented OpenUI layout skeletons** from natural-language prompts, conditioned on a **DESIGN.md** design system. Literal copy is deferred to a separate copy model.

## Goals (this ship cycle)

- Official **`openuiLibrary`** (~54 components) via `@openuidev/react-ui`
- **DESIGN.md** context + `@google/design.md` linter as a composite preference reward
- Harden: HF context default (CLI), larger RICO, remote train, stronger eval
- Cactus export/bench adapter, Awwwards scraping source, DPO/preference stage

## Non-goals (still deferred)

- Consistency distillation
- Full React `@openuidev/react-lang` visual playground
- Custom Cactus NEON kernel authorship in this repo
- Production copy SLM

## Official OpenUI Lang (source of truth)

Parsing, serialization, and system-prompt generation use **`@openuidev/lang-core`** via [`tools/openui_bridge/`](../../tools/openui_bridge/), which re-exports official **`openuiLibrary`**.

Root: `Stack`. Content components include `TextContent`, `Card([children])`, `Button`, `Input`, `ImageBlock`, forms, charts, etc.

### Placeholder policy

User-facing string props (`text`, `label`, `title`, `placeholder`, `alt`, …) **must** be placeholders like `:hero.title`.

### DESIGN.md

- Fixture: [`fixtures/design_md/default.DESIGN.md`](../../fixtures/design_md/default.DESIGN.md)
- Lint bridge: [`tools/design_md_bridge/`](../../tools/design_md_bridge/) (`@google/design.md`)
- Records may carry `design_md`; TwoTower context = `prompt + DESIGN.md`
- Preference reward: grammar → placeholders → linter score → layout metrics

## Model architecture

```
prompt + DESIGN.md → Context Tower (scratch | frozen HF) → hidden states
                                      ↓ cross-attn
         Trainable Denoiser (MaskGIT) → OpenUI tokens
                                      ↓
         streaming parser + placeholder policy
                                      ↓
                         placeholder OpenUI program
```

Optional preference/DPO stage ranks candidates with the composite reward.

## Data sources

| Source | Notes |
| --- | --- |
| RICO | Semantic screens → openuiLibrary (+ default DESIGN.md) |
| Awwwards fixtures | [`fixtures/awwwards/sites.jsonl`](../../fixtures/awwwards/sites.jsonl) |
| Hand fixtures | [`fixtures/train_seeds.jsonl`](../../fixtures/train_seeds.jsonl) |

## Harnesses / CLIs

- `scripts/build_train_data.py` — `--source rico|fixture|awwwards|rico+awwwards|all`
- `scripts/train_model.py` — `--context-backend hf` (default)
- `scripts/train_preference.py` — build-pairs / train
- `scripts/export_cactus.py` / `scripts/bench_cactus.py`
- `scripts/remote_train.py` — SSH pod train + pull
- `scripts/serve_playground.py` — accepts optional `design_md`

## Roadmap status

1–6: prior revisions (done)
7. Full openuiLibrary + DESIGN.md + harden + Cactus/Awwwards/DPO (**this revision**)

### Eval-driven ship gates (current)

| Suite | Gate | Ship (`twotower_v1_ship`) |
| --- | --- | --- |
| smoke | parse ≥ 0.66, struct ≥ 0.35, ph_valid ≥ 0.25, reward ≥ 0.35 | **pass** (1.0 / 1.0 / 0.8 / 0.76) |
| held_out (fixture) | parse ≥ 0.15, struct ≥ 0.25 | **pass** |
| rico_held | diagnostic only | tracked in scoreboard |

Rebuild + scoreboard:

```bash
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/train_data/v1/manifest.json
python -m scripts.evaluate_model --suites smoke,held_out,adversarial,ood,rico_held \
  --train-dir outputs/train_data/v1_fixture_up \
  --test-dir outputs/test_data/v1 \
  --run-id twotower_v1_ship \
  --fail-under-parse-rate 0.66 \
  --fail-under-structural-similarity 0.35 \
  --fail-under-placeholder-validity 0.25 \
  --fail-under-reward-score 0.35
```
