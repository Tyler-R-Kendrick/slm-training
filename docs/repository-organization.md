# Repository organization

Keep one obvious owner for every tracked file. Before adding a path, search with
`rg --files` and `rg` and extend the existing owner when one exists.

## Placement

| Content | Location |
| --- | --- |
| Python implementation | `src/slm_training/` or the existing `src/gpu_multi_farm/` package |
| Runnable entrypoints and maintenance checks | `scripts/` |
| Tests mirroring implementation domains | `tests/` |
| Small committed inputs and expected artifacts | `src/slm_training/resources/` |
| Human-authored design, operations, and measured evidence | `docs/` |
| OpenWiki-generated agent navigation | `docs/openwiki/` |
| Self-contained Node/frontend packages | `src/apps/` |
| Canonical agent skills | `.agents/skills/` |
| Client discovery links and hooks | `.claude/`, `.cursor/`, `.codex/`, `.github/hooks/` |

The repository root is an allowlist for required manifests and cross-agent
instructions. Application code and owned resources belong below `src/`; generated
documentation belongs below `docs/`. Do not add a new root path without
updating this guide and `scripts/repo_policy.py` in the same reviewed change.

## Moves and renames

Use Git for every tracked relocation:

```bash
git mv old/path new/path
rg -n 'old/path|old\.module' .
git diff --summary --find-renames
git log --follow -- new/path
```

Update imports, links, manifests, workflows, generated indexes, and tests in
the same change. Raw `mv` remains fine for ignored outputs and temporary files;
agent hooks block it when a tracked repository path is involved.

## Canonical copies

- Keep each skill only under `.agents/skills/<name>/`.
- Use `../../.agents/skills/<name>` symlinks under `.claude/skills/` and
  `.cursor/skills/`; Codex and Copilot discover `.agents/skills/` directly.
- Keep generated frontend assets, experiment evidence, resources, and vendored
  marketplace skills only where their owning workflow documents them.
- Do not add a second helper, schema, config, or guide for an existing concern;
  extend or relocate the current owner.

## Enforcement

Run the repository policy directly or through the existing changed-file check:

```bash
python -m scripts.repo_policy
.githooks/check-changed
```

The policy rejects unapproved root paths, copied skill mirrors, redundant
Codex skill copies, and newly tracked ignored artifacts. The tracked
pre-commit hook and CI run the same check. Agent `PreToolUse` hooks additionally
reject raw moves of tracked paths; CI remains authoritative because client
hook coverage varies.
