# Agent skills (canonical)

This directory is the **source of truth** for repo skills. Tool-discovery copies
live under `.claude/skills/` and `.cursor/skills/` (full mirrors for
repo-authored skills; symlinks OK for generated ones like `hf-cli`).

| Skill | Purpose |
| --- | --- |
| `documenting-experiment-results` | Update `docs/design/` after every experiment run |
| `honest-ship-eval` | Multi-suite honest ship gates vs fixture demo |
| `running-experiment-matrices` | Quality / grammar / perf / phase matrices |
| `hf-cli` | Hugging Face Hub CLI (`hf`) — models, datasets, jobs, auth |
| `playwright-cli` | Browser / playground automation |

## Sync rules

- **Repo-authored skills:** edit here, then copy into `.claude/skills/` and
  `.cursor/skills/` (keep contents identical).
- **`hf-cli`:** regenerate from the installed CLI (do not hand-edit):

  ```bash
  hf skills add --force
  hf skills add --claude --force
  hf skills add --dest=.cursor/skills --force
  ```

  Prefer leaving `.claude/skills/hf-cli` and `.cursor/skills/hf-cli` as
  symlinks to this directory.

Repo process rules for every agent: [`../../AGENTS.md`](../../AGENTS.md).
