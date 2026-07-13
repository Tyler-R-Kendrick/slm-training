# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation (official `@openuidev/lang-core`), a **TwoTower** masked-diffusion model, plus a **GPU multi-farm MCP**.

## What's included

1. **Training-data harness** — build/validate versioned train corpora
2. **Testing-data harness** — held-out / adversarial / OOD eval suites
3. **Model-building harness** — train/eval with **TwoTower** (default) or stub plug-in
4. **OpenUI Lang bridge** — Node sidecar over official `@openuidev/lang-core`
5. **GPU multi-farm MCP** — list / launch / cost-project across Vast.ai, RunPod, Lambda

See [docs/design/openui-twotower.md](docs/design/openui-twotower.md), [docs/design/research-lineage.md](docs/design/research-lineage.md) (papers → code), [docs/design/research-correction-critics.md](docs/design/research-correction-critics.md) (V4 remask / trust-gate / honest inventory; V6 CoRe/T2M), [docs/design/verifier-guided-repair.md](docs/design/verifier-guided-repair.md) (PDDL-Instruct / verifier-repair applicability map), [docs/design/quality-experiment-matrix.md](docs/design/quality-experiment-matrix.md) (E0–E75 + X0–X8 matrices; E34 deferred), [docs/design/speculative-denoising.md](docs/design/speculative-denoising.md) (V7 stability / dependency-cluster / survival / successor-cache decode), [docs/design/dsl-native-tokenizer.md](docs/design/dsl-native-tokenizer.md) (V5 lexer alphabet), [docs/design/grammar-fastpath.md](docs/design/grammar-fastpath.md), [docs/design/grammar-backends.md](docs/design/grammar-backends.md), [docs/design/structure-only-eval.md](docs/design/structure-only-eval.md), [docs/design/adversarial-review.md](docs/design/adversarial-review.md), [docs/design/runtime-performance.md](docs/design/runtime-performance.md), and [docs/design/gpu-multi-farm-mcp.md](docs/design/gpu-multi-farm-mcp.md).

## Quick start

```bash
# Node.js 20-22 is required for the locked bridge and browser dependencies.
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,hf]"

# Official OpenUI parser + DESIGN.md bridges
cd tools/openui_bridge && npm ci && cd ../..
cd tools/design_md_bridge && npm ci && cd ../..

# optional MCP server deps
pip install -e ".[mcp]"
# optional live RICO download
pip install -e ".[rico]"
```

## Quick start (train / disjoint test)

```bash
# High-quality versioned corpus (default: all sources + quality synthesizer)
python -m scripts.build_train_data --source all --version v1 --synthesizer quality

# Fast fixture-only rebuild
python -m scripts.build_train_data --source fixture --version v0 --synthesizer quality

# Test suites with strict leakage checks against the train manifest
python -m scripts.build_test_data --source both --version v1 \
  --train-manifest outputs/train_data/v1/manifest.json

python -m scripts.train_model \
  --train-dir outputs/train_data/v1 \
  --model twotower \
  --context-backend hf \
  --steps 200 \
  --run-id twotower_v1

python -m scripts.evaluate_model \
  --test-dir outputs/test_data/v1 \
  --model twotower \
  --run-id twotower_v1 \
  --ship-gates
```

Honest ship path (V4 inventory-in-prompt / V6 stacked champion):

```bash
python -m scripts.run_quality_matrix --matrix v4 --only E35,E36 \
  --steps 40 --device cpu --context-backend scratch --no-design-md-context

# V6: CoRe remask + slot-aware trust + honest V5 alphabet
python -m scripts.run_quality_matrix --matrix v6 --only E53 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context
```

Train artifacts land in `outputs/train_data/<version>/` (`records.jsonl`, `manifest.json`, `stats.json`). The flush pipeline: curated seeds + RICO + Awwwards → deterministic quality synth → per-record DESIGN.md + OpenUI validate → quality gates → stable sort by `id` + content fingerprint.

Eval uses **meaningful parse** (rejects empty stacks, missing placeholders, and low gold component-type recall), strict `placeholder_fidelity` for ship gates, `structural_similarity`, and composite `reward_score` (does not credit gold DESIGN.md lint). Suites: smoke/held_out (fixtures), `rico_held`, adversarial, ood. Soft `placeholder_validity` is diagnostic only.

**Fixture demo vs ship:** a tiny upsample + scratch + smoke-only fail-under is wiring only. Readiness requires `--ship-gates` on the full scoreboard (see adversarial review).

Expand `rico_held` with 1500 additional HF RICO screens (cached under `fixtures/rico/hf_test_cache.jsonl`):

```bash
python -m scripts.build_test_data \
  --source both --version v1 \
  --train-manifest outputs/train_data/v1/manifest.json \
  --rico-hf-split test --rico-limit 2600 --target-records 1500
```

```bash
pytest
```

## OpenUI Lang

Fixtures and validation use official **`openuiLibrary`** syntax, e.g.:

```
root = Stack([hero], "column")
hero_title = TextContent(":hero.title")
hero_body = TextContent(":hero.body")
hero = Card([hero_title, hero_body])
```

Content props must be placeholder strings. Parsing/serialization/prompt generation come from `@openuidev/lang-core` + `@openuidev/react-ui` — see [`tools/openui_bridge/`](tools/openui_bridge/).

DESIGN.md conditioning + linter: [`tools/design_md_bridge/`](tools/design_md_bridge/) and [`fixtures/design_md/`](fixtures/design_md/).

## Web playground (annotate)

```bash
pip install -e ".[dev,torch,web]"
python -m scripts.serve_playground --port 8765
# For network exposure, set SLM_ANNOTATION_TOKEN and add --public.
# open http://127.0.0.1:8765
```

The demo checkpoint lives in `fixtures/checkpoints/playground_demo/` (committed
`last.pt` + tokenizer + meta). To regenerate it:

```bash
python -m scripts.bootstrap_playground --force
```

If `last.pt` is missing after a sparse checkout, run the bootstrap command above
before starting the playground.
Annotate mode (default UI): auto-generated prompts, prefetch 1–2 samples ahead, and a live **OpenUI visual preview** (same `@openuidev/react-lang` `Renderer` path as [openui.com/demo](https://www.openui.com/demo/github)).

| Input | Action |
|-------|--------|
| `↑` | Thumbs up (persist, stay on sample) |
| `↓` | Thumbs down (persist, stay on sample) |
| `←` / `→` | Previous / next sample |
| typing | Focus optional note |
| swipe | Mobile: horizontal navigate, vertical grade |

Annotations append to `outputs/annotations/feedback.jsonl`. Invalid model outputs are quarantined to `outputs/annotations/bad_outputs.jsonl` (never shown in the app). Thumbs-up rows promote into `fixtures/annotations/human_train.jsonl` (merged by `build_train_data`). Opposite ratings on the same prompt also write `outputs/preferences/human_pairs.jsonl`.

```bash
python -m scripts.export_annotations status
python -m scripts.export_annotations export
```

### Rebuild the OpenUI preview bundle

```bash
npm run preview:install
npm run preview:build
# writes src/slm_training/web/static/preview/{preview.js,preview.css}
```

### Playwright visual / e2e

```bash
npm ci
npx playwright install chromium
# optional agent skills (already in .agents/skills + .cursor/skills)
playwright-cli install --skills
npm run test:e2e
```

MCP (Cursor): [`.cursor/mcp.json`](.cursor/mcp.json) launches `@playwright/mcp`.


- **Context tower**: scratch TokenEncoder **or** frozen HF model (`--context-backend hf`, default `HuggingFaceTB/SmolLM2-135M`)
- **Denoiser tower**: MaskGIT-style masked token prediction with cross-attention to context ([Chang et al. 2022](https://arxiv.org/abs/2202.04200); adapted)
- **Grammar decode**: DFA force-emit + MaskGIT hole-admit + LTR certify so constrained samples stay valid OpenUI ([research lineage](docs/design/research-lineage.md); `--no-grammar` to disable)
- **Output tokenizer**: dual-mode — default **compositional** `OpenUITokenizer`, or V5 **lexer / DSL-native** `DSLNativeTokenizer` (`output_tokenizer=lexer`; see [dsl-native-tokenizer.md](docs/design/dsl-native-tokenizer.md))
- **Eval**: `parse_rate` via lang-core, placeholder fidelity, canonical tree match — no gold oracle at generate time

```bash
# Optional HF context (requires: pip install -e ".[hf]")
python -m scripts.train_model --model twotower --context-backend hf \
  --hf-model HuggingFaceTB/SmolLM2-135M --steps 200 --run-id twotower_hf
```

## GPU multi-farm MCP

```bash
cp .env.example .env
pip install -e ".[mcp]"
GPU_MULTI_FARM_MODE=mock python -m scripts.multi_farm_mcp
```

## Layout

```
src/slm_training/
  dsl/                 # OpenUI adapter, schema, parser, production codec
  data/                # RICO / Awwwards adapters + leakage fingerprints
  models/              # TwoTower, grammar_diffusion, tokenizers, remask
  harnesses/           # train_data, test_data, model_build
  grammar_fastpath/    # DFA force-emit, MaskGIT admit, FastPathGate
  grammar_backends/    # lang-core / Lark / hybrid / toy-layout backends
  preference/          # composite reward + DPO surrogate
  rl/                  # GRPO-lite
  quality/             # curriculum stages
  retrieval/           # nearest-skeleton helpers
  web/                 # annotate playground API + static preview
  design_md/           # DESIGN.md Python wrapper
  annotations/         # human feedback helpers
  accel/               # AMP / torch.compile
  telemetry/           # cycle timing
  cactus/              # export + NEON kernel sketches
  compression/         # BF16 codebook sidecar
src/gpu_multi_farm/    # FastMCP server + farm adapters
tools/openui_bridge/   # @openuidev/lang-core Node sidecar
tools/design_md_bridge/
tools/openui_preview/
scripts/               # CLIs
fixtures/              # seed pairs + RICO semantic slices
docs/design/           # architecture + research lineage + contracts
```
