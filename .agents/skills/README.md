# Agent skills (canonical)

This directory is the **source of truth** for repo skills. Tool-discovery
entries under `.claude/skills/`, `.cursor/skills/`, and `.grok/skills/` are
symlinks back here.

| Client | Discovery path |
| --- | --- |
| Claude Code | `.claude/skills/` (symlinks) |
| Cursor | `.cursor/skills/` (symlinks) |
| Grok Build | `.grok/skills/` (symlinks) + scans `.agents/skills/` |
| Codex | `.agents/skills/` directly — **no** `.codex/skills/` tree |
| GitHub Copilot | `.agents/skills/` directly |
| Gemini | `.agents/skills/` (via `GEMINI.md`) |

## Repo-authored

| Skill | Purpose |
| --- | --- |
| `documenting-experiment-results` | Update `docs/design/` + `MODEL_CARD.md` / README summary after experiments & checkpoints |
| `dashboard-openui-parity` | Keep each dashboard page's interpreted-mode `static/openui/*.openui` program at parity with its compiled React page |
| `honest-ship-eval` | Multi-suite honest ship gates vs fixture demo |
| `running-experiment-matrices` | Quality / grammar / perf / phase matrices |
| `openui-autoresearch` | Evidence-grounded research, hypothesis matrices, feedback, execution, and RL readiness |
| `improve-openui-harnesses` | Harness-family owners, invariants, outputs, improvement checks, and anti-sprawl rules |
| `improve-lean-optimums` | Diagnose certified metric-band misses and improve the correct harness, model, Lean calculation, or assumption |
| `autotrain` | Continuous local model+harness improvement by default; explicit phases/`--once` are finite |
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

Refresh. **Run the whole block**, including the normalisation step — the
marketplace installer copies rather than symlinks, and `-a codex` writes a
`.codex/skills/` tree that Codex does not need. Both states are rejected by
`python -m scripts.repo_policy`, so the installer alone leaves the repo failing.

```bash
# Note: no `-a codex` — Codex and GitHub Copilot read .agents/skills/ directly,
# and scripts/repo_policy.py rejects a redundant .codex/skills/ copy.
npx skills add DietrichGebert/ponytail --skill '*' \
  -a claude-code -a cursor -a github-copilot -y --copy
npx skills add JuliusBrussee/caveman \
  --skill caveman --skill caveman-commit --skill caveman-review \
  --skill caveman-help --skill caveman-compress --skill caveman-stats \
  -a claude-code -a cursor -a github-copilot -y --copy
npx skills add roman-ryzenadvanced/headroom-skill --skill headroom \
  -a claude-code -a cursor -a github-copilot -y --copy

# Normalise discovery roots: every canonical skill gets a symlink, no copies,
# no stray Codex tree. Idempotent, and also fixes an `hf skills add --dest=`
# copy. repo_policy checks both directions, so a missing mirror fails too.
rm -rf .codex/skills
mkdir -p .claude/skills .cursor/skills .grok/skills
for name in $(ls .agents/skills); do
  [ -d ".agents/skills/$name" ] || continue
  for root in .claude/skills .cursor/skills .grok/skills; do
    rm -rf "$root/$name"
    ln -s "../../.agents/skills/$name" "$root/$name"
  done
done
python -m scripts.repo_policy   # skill mirrors must be clean (other WIP may fail)

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

Refresh. `hf skills add --dest=…` copies rather than symlinks, so re-run the
normalisation loop from the token-efficiency block above (and
`python -m scripts.repo_policy`) after any of these.

```bash
hf skills update
hf skills add --force                 # regenerate hf-cli
hf skills add <name> --force          # one skill
hf skills add --claude --force        # Claude symlinks
# then: normalisation loop + python -m scripts.repo_policy
```

Full HF-context trains sync checkpoints to `hf://buckets/TKendrick/OpenUI` (see `docs/design/checkpoint-bucket.md`).

Repo process rules: [`../../AGENTS.md`](../../AGENTS.md).
