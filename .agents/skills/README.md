# Agent skills (canonical)

This directory is the **source of truth** for repo skills. Tool-discovery
entries under `.claude/skills/` and `.cursor/skills/` are symlinks back here.

Codex and GitHub Copilot also load project skills from **`.agents/skills/`**.

## Repo-authored

| Skill | Purpose |
| --- | --- |
| `documenting-experiment-results` | Update `docs/design/` + `MODEL_CARD.md` / README summary after experiments & checkpoints |
| `dashboard-openui-parity` | Keep each dashboard page's interpreted-mode `static/openui/*.openui` program at parity with its compiled React page |
| `honest-ship-eval` | Multi-suite honest ship gates vs fixture demo |
| `running-experiment-matrices` | Quality / grammar / perf / phase matrices |
| `openui-autoresearch` | Evidence-grounded research, hypothesis matrices, feedback, execution, and RL readiness |
| `improve-openui-harnesses` | Harness-family owners, invariants, outputs, improvement checks, and anti-sprawl rules |
| `autotrain` | Facade for running any training pipeline phase; per-phase `references/*.md` load on demand |
| `autoresearch` | Knowledge-driven research loop: read/update repo + personal brains (OpenWiki / OKF / Obsidian), prior-work discovery, autotrain hypothesis loop, and Linear issue/milestone/project emission; per-stage `references/*.md` load on demand |
| `playwright-cli` | Browser / playground automation |
| `frontier-describe` | Train-only frozen paraphrase / ladder / edit / vision artifacts |
| `organize-repository` | Canonical file placement, deduplication, and `git mv` workflow |
| `rtk` | Prefer Rust Token Killer for verbose shell output ([`RTK.md`](../../RTK.md)) |

Edit only here; discovery symlinks update every client automatically.

## Token-efficiency pack

Pinned via root [`skills-lock.json`](../../skills-lock.json). Installed for
**claude-code**, **cursor**, **codex**, and **github-copilot**.

| Skill | Source |
| --- | --- |
| `ponytail` (+ review/audit/debt/gain/help) | [DietrichGebert/ponytail](https://github.com/DietrichGebert/ponytail) |
| `caveman` (+ commit/review/help/compress/stats) | [JuliusBrussee/caveman](https://github.com/JuliusBrussee/caveman) |
| `headroom` (+ `scripts/` helpers) | [roman-ryzenadvanced/headroom-skill](https://github.com/roman-ryzenadvanced/headroom-skill) |

Cursor always-on / opt-in rules: [`.cursor/rules/`](../../.cursor/rules/).
GHCP: [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md).

Refresh:

```bash
npx skills add DietrichGebert/ponytail --skill '*' \
  -a claude-code -a cursor -a codex -a github-copilot -y --copy
npx skills add JuliusBrussee/caveman \
  --skill caveman --skill caveman-commit --skill caveman-review \
  --skill caveman-help --skill caveman-compress --skill caveman-stats \
  -a claude-code -a cursor -a codex -a github-copilot -y --copy
npx skills add roman-ryzenadvanced/headroom-skill --skill headroom \
  -a claude-code -a cursor -a codex -a github-copilot -y --copy
# Prefer symlinks for discovery dirs (after --copy duplicates):
for name in ponytail ponytail-audit ponytail-debt ponytail-gain ponytail-help \
  ponytail-review caveman caveman-commit caveman-compress caveman-help \
  caveman-review caveman-stats headroom rtk; do
  rm -rf ".claude/skills/$name" ".cursor/skills/$name"
  ln -s "../../.agents/skills/$name" ".claude/skills/$name"
  ln -s "../../.agents/skills/$name" ".cursor/skills/$name"
done
# Re-copy headroom helpers if the skills CLI only dropped SKILL.md:
# git clone --depth 1 https://github.com/roman-ryzenadvanced/headroom-skill /tmp/hr
# cp -a /tmp/hr/{scripts,prompts,docs,examples,AGENTS.md,CLAUDE.md,LICENSE,NOTICE} .agents/skills/headroom/
```

RTK binary: see [`RTK.md`](../../RTK.md).

## Hugging Face ([huggingface/skills](https://github.com/huggingface/skills))

Installed into this directory with `hf skills add` (Cursor guidance: marketplace
ships `hf-cli`; use the CLI for the rest). Symlinked under `.claude/skills/` and
`.cursor/skills/`.

| Skill | Notes |
| --- | --- |
| `hf-cli` | Generated from local CLI (`hf skills add --force`) |
| `hf-mem` | Model memory estimation |
| `huggingface-best` | Best/recommended model discovery |
| `huggingface-community-evals` | inspect-ai / lighteval |
| `huggingface-datasets` | Dataset Viewer API |
| `huggingface-gradio` | Gradio UIs |
| `huggingface-llm-trainer` | TRL / Unsloth + HF Jobs |
| `huggingface-local-models` | llama.cpp / GGUF local |
| `huggingface-lora-space-builder` | LoRA → Spaces demo |
| `huggingface-paper-publisher` | Publish papers on the Hub |
| `huggingface-papers` | Papers API / pages |
| `huggingface-spaces` | Spaces deploy / ZeroGPU |
| `huggingface-tool-builder` | HF API tooling |
| `huggingface-trackio` | Trackio experiment tracking |
| `huggingface-vision-trainer` | Vision train/fine-tune on Jobs |
| `huggingface-zerogpu` | ZeroGPU demos |
| `train-sentence-transformers` | Sentence Transformers train |
| `transformers-js` | Transformers.js |
| `trl-training` | TRL language-model training |
| `hf-cloud-*` | SageMaker / AWS helper skills |

Refresh:

```bash
hf skills update
hf skills add --force                 # regenerate hf-cli
hf skills add <name> --force          # one skill
hf skills add --claude --force        # Claude symlinks
```

Full HF-context trains sync checkpoints to `hf://buckets/TKendrick/OpenUI` (see `docs/design/checkpoint-bucket.md`).

Repo process rules: [`../../AGENTS.md`](../../AGENTS.md).
