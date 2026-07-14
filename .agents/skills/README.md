# Agent skills (canonical)

This directory is the **source of truth** for repo skills. Tool-discovery copies
live under `.claude/skills/` and `.cursor/skills/` (full mirrors for
repo-authored skills; **symlinks** for Hugging Face marketplace / generated
skills).

## Repo-authored

| Skill | Purpose |
| --- | --- |
| `documenting-experiment-results` | Update `docs/design/` after every experiment run |
| `honest-ship-eval` | Multi-suite honest ship gates vs fixture demo |
| `running-experiment-matrices` | Quality / grammar / perf / phase matrices |
| `playwright-cli` | Browser / playground automation |

Edit here, then copy into `.claude/skills/` and `.cursor/skills/`.

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
