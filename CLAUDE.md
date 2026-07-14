# Claude Code instructions

Follow **[AGENTS.md](AGENTS.md)** — the canonical instructions for every coding
agent in this repo.

Load skills from `.agents/skills/` (mirrored / symlinked under `.claude/skills/`).
After any train / eval / benchmark / matrix run, use
`documenting-experiment-results`. For Hub / HF work, use `hf-cli`
(`hf skills add --claude --force` to refresh).
