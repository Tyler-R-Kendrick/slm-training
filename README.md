# slm-training

Novel SLM experiments: harnesses for **placeholder OpenUI** layout generation (official `@openuidev/lang-core`), a **TwoTower** masked-diffusion model, plus a **GPU multi-farm MCP**.

## What's included

1. **Training-data harness** — build/validate versioned train corpora
2. **Testing-data harness** — held-out / adversarial / OOD eval suites
3. **Model-building harness** — lineage-first **TwoTower** and causal-LoRA tracks
4. **OpenUI Lang bridge** — Node sidecar over official `@openuidev/lang-core`
5. **GPU multi-farm MCP** — list / launch / cost-project across Vast.ai, RunPod, Lambda

See [docs/design/model-lineage.md](docs/design/model-lineage.md) (canonical two-track cycle), [docs/design/openui-twotower.md](docs/design/openui-twotower.md), [docs/design/research-lineage.md](docs/design/research-lineage.md) (papers → code), [docs/design/research-correction-critics.md](docs/design/research-correction-critics.md) (V4 remask / trust-gate / honest inventory; V6 CoRe/T2M), [docs/design/verifier-stack.md](docs/design/verifier-stack.md) (G0–G12 corpus gates + confidence tiers), [docs/design/abstraction-house-style.md](docs/design/abstraction-house-style.md) (L0–L5 determinacy, grounding, and canonical defaults), [docs/design/verifier-guided-repair.md](docs/design/verifier-guided-repair.md) (PDDL-Instruct / verifier-repair applicability map), [docs/design/quality-experiment-matrix.md](docs/design/quality-experiment-matrix.md) (E0–E75 + X0–X8 matrices; E34 deferred), [docs/design/speculative-denoising.md](docs/design/speculative-denoising.md) (V7 stability / dependency-cluster / survival / successor-cache decode), [docs/design/dsl-native-tokenizer.md](docs/design/dsl-native-tokenizer.md) (V5 lexer alphabet), [docs/design/grammar-fastpath.md](docs/design/grammar-fastpath.md), [docs/design/grammar-backends.md](docs/design/grammar-backends.md), [docs/design/structure-only-eval.md](docs/design/structure-only-eval.md), [docs/design/adversarial-review.md](docs/design/adversarial-review.md), [docs/design/runtime-performance.md](docs/design/runtime-performance.md), [docs/design/hf-jobs-train.md](docs/design/hf-jobs-train.md) (HF Jobs full train — not ZeroGPU), [docs/design/gpu-multi-farm-mcp.md](docs/design/gpu-multi-farm-mcp.md), and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

## Model card (summary)

Full card: **[docs/MODEL_CARD.md](docs/MODEL_CARD.md)**. Agents update both this
summary and the full card whenever a checkpoint is created or promoted.

| Role | Checkpoint | Where | Claim |
| --- | --- | --- | --- |
| Playground demo | `playground_demo/last.pt` | `fixtures/checkpoints/playground_demo/` (git) | Wiring / annotate UI only |
| Restructure CPU verify | `restructure_cpu_scratch_v0/last.pt` | `outputs/runs/…` (local) | Fixture scratch train OK; smoke parse 0.0 — not ship |
| Local DirectML verify | `local_directml_adreno_20260714/last.pt` | `outputs/runs/…` (local) | Adreno GPU train/checkpoint OK; 5-step wiring run, not evaluated or ship |
| Matrix honest champion | V6 E53 family | `outputs/runs/` + matrix docs | Scratch + limited `rico_held` — not production HF ship |
| P13 matched E50 controls | fixture + integrated E50 | `/tmp/slm17-e50-*-honest/` (local scratch) | Integrated fidelity +0.04 held / +0.0333 RICO; parse 0.0, not ship |
| Production HF ship | *(none yet)* | [HF Bucket `TKendrick/OpenUI`](https://huggingface.co/buckets/TKendrick/OpenUI) `checkpoints/<run_id>/` | Register here after first full HF sync + `--ship-gates` |

**Load demo:** `python -m scripts.serve_playground` · **Full train sync:** set
`HF_TOKEN`, then `train_model --context-backend hf` (auto-uploads). Details,
eval tables, and history live in the model card.

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

# Full HF-context trains sync checkpoints to the OpenUI bucket
# (https://huggingface.co/buckets/TKendrick/OpenUI). Requires HF_TOKEN.
export HF_TOKEN=hf_...   # or: hf auth login
python -m scripts.train_model \
  --train-dir outputs/train_data/v1 \
  --model twotower \
  --context-backend hf \
  --steps 200 \
  --run-id twotower_v1
# → hf://buckets/TKendrick/OpenUI/checkpoints/twotower_v1/

python -m scripts.evaluate_model \
  --test-dir outputs/test_data/v1 \
  --model twotower \
  --run-id twotower_v1 \
  --ship-gates
```

Local-only / CI scratch: add `--no-sync-checkpoints` (matrix scripts default to
scratch and stay local). Manual sync:
`python -m scripts.sync_checkpoints --run-dir outputs/runs/<id> --ensure-bucket`.
See [docs/design/checkpoint-bucket.md](docs/design/checkpoint-bucket.md).

Honest ship path (V4 inventory-in-prompt / V6 stacked champion):

```bash
python -m scripts.run_quality_matrix --matrix v4 --only E35,E36 \
  --steps 40 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control

# V6: CoRe remask + slot-aware trust + honest V5 alphabet
python -m scripts.run_quality_matrix --matrix v6 --only E53 \
  --steps 80 --device cpu --context-backend scratch --no-design-md-context \
  --scratch-control
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

## Mission Control dashboard

`serve_playground` serves a **control-plane + observability SPA** at `/` — one
pane of glass over the whole lifecycle (data → experiments → smoke →
checkpoints/promotion) — plus the classic annotate playground at `/playground`.

```bash
pip install -e ".[dev,torch,web]"
python -m scripts.serve_playground --port 8765        # full control plane (local)
python -m scripts.serve_playground --no-enable-jobs   # read-only observability
# For network exposure, set SLM_ANNOTATION_TOKEN and add --public.
# open http://127.0.0.1:8765
```

Surfaces (React 19 + Vite SPA, dark-first "mission control" design system):

| Route | What |
| --- | --- |
| `/` Overview | Live jobs, experiment scoreboard, checkpoint roster, corpus health, system status, **remote dispatches** |
| `/data` | Navigate + generate versioned corpora (`build_train_data` / `build_test_data`) |
| `/experiments` | Quality / grammar / perf / phase matrices; run `run_*_matrix`; **dispatch full GPU trains** (`hf_jobs_train` / `remote_train`); drill into any run |
| `/smoke` | Smoke canary + perf & telemetry; launch wiring runs |
| `/checkpoints` | Roster + **live configurable ship gates** + promote / deploy + blinded A/B |
| `/runs/<id>` | Per-run detail — gate matrix, telemetry spans, `train_summary` metrics, durable-checkpoint link |
| `/playground` | Annotate UI (React); classic vanilla page kept at `/playground/classic` |

**Read vs execute.** Observability views are pure reads (work on a fresh checkout
and on read-only Vercel, falling back to committed `docs/design/*.json` /
`MODEL_CARD.md` / `fixtures/`, tagged with `provenance`). Generate/run/promote
actions execute an **allowlisted** set of scripts as tracked background jobs with
live SSE logs — only when served locally (`--enable-jobs`, default on); Vercel
degrades to read-only automatically. Gate math (`POST /api/gates/evaluate`) is
pure, so the threshold editor stays live even read-only. Backend:
`src/slm_training/web/{observability,jobs,capabilities,routes}.py`; SPA source in
[`tools/dashboard/`](tools/dashboard/) (built bundle committed under
`web/static/app/`, like the preview lib).

## Annotate playground (`/playground`)

```bash
python -m scripts.serve_playground --port 8765
# open http://127.0.0.1:8765/playground
```

`/playground` is the React annotate UI inside the SPA shell (shares the dark
design system); the original standalone vanilla page is preserved at
`/playground/classic`. Both drive the same `/api/sample` + `/api/annotate` flow.

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

### Rebuild the dashboard bundle

```bash
npm run dashboard:install
npm run dashboard:build
# writes src/slm_training/web/static/app/ (built SPA, committed like the preview lib)
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
  --hf-model HuggingFaceTB/SmolLM2-135M --steps 200 --run-id twotower_hf --fast-train
```

## Hugging Face Jobs (full GPU train)

ZeroGPU Spaces are for short demos only. Full trains use managed Jobs:

```bash
python -m scripts.hf_jobs_train --dry-run --run-id twotower_jobs_v1 --steps 200
# submit: export HF_TOKEN=… && python -m scripts.hf_jobs_train --run-id … --steps 200
```

Details: [docs/design/hf-jobs-train.md](docs/design/hf-jobs-train.md).

## GPU multi-farm MCP

```bash
cp .env.example .env
pip install -e ".[mcp]"
GPU_MULTI_FARM_MODE=mock python -m scripts.multi_farm_mcp
```

## Agent instructions

All coding agents (Cursor, Claude Code, Codex, Gemini, Copilot / GHCP, …) must
follow **[AGENTS.md](AGENTS.md)**. Canonical skills live in
[`.agents/skills/`](.agents/skills/) (mirrored under `.claude/skills/` and
`.cursor/skills/`).

**Iron law:** after any train / eval / bench / profile / telemetry / matrix /
reproduction (or decision-informing ad-hoc) run, update `docs/design/` JSON
**and** the matching measured-results markdown. Full trigger list and recipe
checklist: [AGENTS.md](AGENTS.md) (skill: `documenting-experiment-results`).
Do not leave results only under `outputs/`.

### Token-efficiency stack

Repo ships **ponytail**, **caveman**, **headroom**, and **rtk** under
`.agents/skills/` (plus [`RTK.md`](RTK.md), Cursor rules, and GHCP
`.github/copilot-instructions.md`). Details and refresh commands:
[AGENTS.md — Token-efficiency stack](AGENTS.md).

```bash
# RTK binary (once per machine) — must pass `rtk gain`
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
```

### OpenWiki (code mode)

Repository wiki for agents lives under [`openwiki/`](openwiki/) (start at
[`openwiki/quickstart.md`](openwiki/quickstart.md)). Setup uses
[langchain-ai/openwiki](https://github.com/langchain-ai/openwiki) code mode:
[`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md) OpenWiki snippets and
[`.github/workflows/openwiki-update.yml`](.github/workflows/openwiki-update.yml).

```bash
npm install -g openwiki
# needs OPENWIKI_PROVIDER + provider API key in env or ~/.openwiki/.env
openwiki code --update --print
```

Add repo secret `OPENROUTER_API_KEY` (default workflow provider) to enable the
scheduled OpenWiki update PRs.

### Hugging Face CLI + skills

Agents use the official `hf` CLI and the
[huggingface/skills](https://github.com/huggingface/skills) pack (skill:
`hf-cli` plus datasets / papers / trainers / Spaces / … under
[`.agents/skills/`](.agents/skills/)). Cursor also gets the Hugging Face MCP
server via [`.cursor/mcp.json`](.cursor/mcp.json).

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
hf skills add --force
hf skills update
hf skills add --claude --force
hf skills add --dest=.cursor/skills --force
```

Optional Cursor UI: [marketplace — Hugging Face](https://cursor.com/marketplace/huggingface).
CLI docs: [huggingface_hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
Tokens: [settings/tokens](https://huggingface.co/settings/tokens).

### Serena MCP

Semantic code tools via [Serena](https://github.com/oraios/serena) (not
marketplace installs). Project is initialised under [`.serena/`](.serena/);
Cursor / Claude / VS Code MCP configs are wired in-repo. See
[AGENTS.md — Serena MCP](AGENTS.md).

```bash
uv tool install -p 3.13 serena-agent
serena init
serena project health-check
```

## Layout

```
AGENTS.md              # cross-tool agent instructions (required reading)
RTK.md                 # Rust Token Killer usage (shell output compression)
docs/MODEL_CARD.md     # checkpoint roster + eval (README holds a summary)
.agents/skills/        # canonical agent skills
src/slm_training/
  dsl/                 # OpenUI adapter + design_md + grammar/{backends,fastpath}
  harnesses/           # train_data, test_data, model_build, rl, preference,
                       # distill, quality(+retrieval), experiments, annotations
  models/              # TwoTower, grammar_diffusion, tokenizers, remask
  data/                # RICO / Awwwards adapters + leakage fingerprints
  evals/               # loss suites / denoising NLL
  runtime/             # accel, telemetry, compression, cactus
  web/                 # mission-control API (observability + jobs) + annotate playground + SPA
src/gpu_multi_farm/    # FastMCP server + farm adapters
tools/openui_bridge/   # @openuidev/lang-core Node sidecar
tools/design_md_bridge/
tools/openui_preview/
scripts/               # CLIs
fixtures/              # seed pairs + RICO semantic slices
docs/design/           # architecture + research lineage + contracts
tests/
  test_dsl/            # parser, grammar, design_md
  test_harnesses/      # mirrors harnesses/* (rl is its own suite)
  test_runtime/        # accel / cactus / compression
  test_models/ test_data/ test_web/ ...
```
