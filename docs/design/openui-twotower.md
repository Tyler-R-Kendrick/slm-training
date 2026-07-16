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
- Full chat-shell AgentInterface / OpenUIChat (annotate uses a focused `Renderer` island instead)
- Custom Cactus NEON kernel authorship in this repo
- Production copy SLM
- PDDL / VAL / classical planners (see [verifier-guided-repair.md](verifier-guided-repair.md))

## Official OpenUI Lang (source of truth)

Parsing, serialization, and system-prompt generation use **`@openuidev/lang-core`** via [`src/apps/openui_bridge/`](../../src/apps/openui_bridge/), which re-exports official **`openuiLibrary`**.

The pinned `@openuidev/lang-core@0.2.9` bridge and in-process grammar can parse
the prototype v0.5 surface: `$state`, `Query`, `Mutation`, `Action`, tool calls,
objects, member/index access, and expressions. They retain runtime sidecars and
preserve validated whole-program source when canonical tree serialization would
drop declarations. This is parser/codec capability only: the honest layout SLM
training contract remains the published 0.2.x subset and the verifier continues
to reject runtime/dataflow syntax from layout-only datasets.

The canonical 16-hex contract identity remains
[`language_contract.py`](../../src/slm_training/dsl/language_contract.py). Its
grammar and tokenizer hashes change with this surface, so tokenizer-v1 artifacts
must be regenerated for the v2 typed state/builtin alphabet; the
`ExampleRecord` wire shape remains unchanged and stores identity in metadata.

Root: `Stack`. Content components include `TextContent`, `Card([children])`, `Button`, `Input`, `ImageBlock`, forms, charts, etc.

### Placeholder policy

User-facing string props (`text`, `label`, `title`, `placeholder`, `alt`, …) **must** be placeholders like `:hero.title`.

### DESIGN.md

- Fixture: [`src/slm_training/resources/design_md/default.DESIGN.md`](../../src/slm_training/resources/design_md/default.DESIGN.md)
- Lint bridge: [`src/apps/design_md_bridge/`](../../src/apps/design_md_bridge/) (`@google/design.md`)
- Records may carry `design_md`; TwoTower context = `prompt + DESIGN.md`
- Preference reward: grammar → placeholders → linter score → layout metrics
- Eval must not credit gold DESIGN.md lint when `design_md_in_context=false`

## Model architecture

```
prompt + DESIGN.md → Context Tower (scratch | frozen HF) → hidden states
                                      ↓ cross-attn
         Trainable Denoiser (MaskGIT) → OpenUI tokens
                                      ↓
         DFA force-emit / admit + streaming parser + placeholder policy
                                      ↓
                         placeholder OpenUI program
```

Optional preference stage ranks candidates with the composite reward.
**Note:** current “DPO” training is reference-free (surrogate on masked log-probs) — not textbook DPO.

**Papers / techniques → code:** see [research-lineage.md](research-lineage.md)
(MaskGIT, constrained diffusion LLMs, speculative/force-emit, DPO/GRPO surrogates;
verifier-guided repair Adjacent lineage).
Grammar decode details: [grammar-fastpath.md](grammar-fastpath.md).
Online structure-aware masking, insert/delete canvases, and target-length
prediction: [diffusion-data-adapter.md](diffusion-data-adapter.md).
Applicability of PDDL-Instruct-style ideas: [verifier-guided-repair.md](verifier-guided-repair.md).

## Data sources

| Source | Notes |
| --- | --- |
| RICO | Semantic screens → openuiLibrary (+ default DESIGN.md) |
| Awwwards fixtures | [`src/slm_training/resources/awwwards/sites.jsonl`](../../src/slm_training/resources/awwwards/sites.jsonl) |
| Hand fixtures | [`src/slm_training/resources/train_seeds.jsonl`](../../src/slm_training/resources/train_seeds.jsonl) |

Leakage checks use exact + **structural** OpenUI fingerprints (placeholder/binder normalized).

## Harnesses / CLIs

- `scripts/build_train_data.py` — `--source rico|fixture|awwwards|rico+awwwards|all`
- `scripts/train_model.py` — `--context-backend hf` (default)
- `scripts/train_preference.py` — build-pairs / train (reference-free)
- `scripts/export_cactus.py` / `scripts/bench_cactus.py`
- `scripts/remote_train.py` — SSH pod train + pull (trains the `v1` corpus it builds)
- `scripts/serve_playground.py` — accepts optional `design_md`

## Roadmap status

1–6: prior revisions (done)
7. Full openuiLibrary + DESIGN.md + harden + Cactus/Awwwards/preference (**done**)
8. Adversarial review remediations — see [adversarial-review.md](adversarial-review.md) (**done**)
9. Runtime simplify/optimize (PyTorch path); Cactus/NEON kernel stays separate — [runtime-performance.md](runtime-performance.md) (**done**)
10. Quality experiment matrix (all levers) — [quality-experiment-matrix.md](quality-experiment-matrix.md) (**done**; E0–E55 + X0–X8)
11. Accelerator / parallel decode — [accel-parallel.md](accel-parallel.md) (**done**)
12. V4 critic remask / trust gate / honest inventory (E30–E36; E34 deferred) — [research-correction-critics.md](research-correction-critics.md) (**done**; E35/E36 fixture ship)
13. V5 DSL-native / lexer tokenizer (E40–E46) — [dsl-native-tokenizer.md](dsl-native-tokenizer.md) (**done**)
14. V6 CoRe remask / T2M / slot-aware trust / honest champion (E50–E55) — [quality-experiment-matrix.md](quality-experiment-matrix.md) (**done**; E53/E55 fixture ship)
15. Remaining verifier-guided repair gaps (proposed E60–E65: differential validation, failure-cone remask, minimal hard negatives, calibration, trajectory-aligned RL, schema generalization) — [verifier-guided-repair.md](verifier-guided-repair.md) (**proposed**; docs only)

### Eval-driven ship gates (honest policy)

Prior soft gates that declared `twotower_v1_ship` a pass (smoke-only / weak held_out) are **invalid**.

| Suite | Gate | Role |
| --- | --- | --- |
| smoke | parse ≥ 0.66, struct ≥ 0.35, **fidelity** ≥ 0.25, reward ≥ 0.30 | Canary only |
| held_out | parse ≥ 0.40, struct ≥ 0.30, fidelity ≥ 0.15 | Fixture generalization |
| adversarial / ood | parse ≥ 0.25, struct ≥ 0.25 | Stress |
| rico_held | parse ≥ 0.10, struct ≥ 0.20 | Distribution shift (full suite size) |

```bash
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/data/train/v1/manifest.json
python -m scripts.evaluate_model --ship-gates \
  --train-dir outputs/data/train/v1 \
  --test-dir outputs/data/eval/v1 \
  --run-id <run>
```

**Fixture demo** (wiring only): tiny upsample + scratch + smoke fail-unders — not a ship claim.
**Ship candidate**: train `v1`, HF + DESIGN.md when claimed, `--ship-gates` on the full scoreboard.
