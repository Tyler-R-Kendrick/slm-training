# Gemini CLI instructions

Follow **[AGENTS.md](AGENTS.md)** — the canonical instructions for every coding
agent in this repo.

Activate skills from `.agents/skills/`. After any train / eval / benchmark /
matrix run, use `documenting-experiment-results`. Token stack: `ponytail`,
`caveman`, `headroom`, `rtk` (see `AGENTS.md` / `RTK.md`). Hugging Face pack:
`hf-cli` + marketplace skills from
[huggingface/skills](https://github.com/huggingface/skills)
(`hf skills add --force` / `hf skills update` to refresh).
When a checkpoint is created or promoted, update `docs/MODEL_CARD.md` and the README model-card summary.
