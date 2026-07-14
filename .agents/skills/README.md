# Agent skills (canonical)

This directory is the **source of truth** for repo skills. Copies under
`.claude/skills/` and `.cursor/skills/` must stay identical.

| Skill | Purpose |
| --- | --- |
| `documenting-experiment-results` | Update `docs/design/` after every experiment run |
| `honest-ship-eval` | Multi-suite honest ship gates vs fixture demo |
| `running-experiment-matrices` | Quality / grammar / perf / phase matrices |
| `playwright-cli` | Browser / playground automation |

When editing a skill, update all three trees (or copy from here). Repo process
rules that apply to every agent live in `/AGENTS.md`.
