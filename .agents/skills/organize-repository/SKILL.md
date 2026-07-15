---
name: organize-repository
description: Keep this repository's tracked files and directories canonical, navigable, and free of redundant copies. Use before creating, moving, renaming, deleting, or duplicating tracked files or folders; when adding modules, scripts, docs, fixtures, tools, skills, hooks, or top-level paths; and when reviewing repository sprawl or placement. Require git mv for tracked relocations.
---

# Organize Repository

1. Read `docs/repository-organization.md` before choosing a destination.
2. Search existing paths and symbols with `rg --files` and `rg` before adding a file. Extend an existing owner when it already covers the responsibility.
3. Keep implementation in `src/`, entrypoints in `scripts/`, tests in `tests/`, fixtures in `src/slm_training/resources/`, durable docs in `docs/`, generated agent wiki pages in `docs/openwiki/`, and self-contained frontend/Node packages in `src/apps/`.
4. Use `git mv <old> <new>` for every tracked relocation. Update imports, links, manifests, workflows, and tests in the same change.
5. Keep `.agents/skills/` canonical. Discovery entries under `.claude/skills/` and `.cursor/skills/` must be symlinks; do not copy skills into `.codex/skills/`.
6. Run `python -m scripts.repo_policy` and the tests selected by `.githooks/check-changed`.
7. Verify moves with `git diff --summary --find-renames` and inspect history with `git log --follow -- <path>`.

Do not create a new top-level path without updating the organization guide and policy checker in the same reviewed change.
