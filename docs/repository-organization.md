# Repository organization

Keep one obvious owner for every tracked file. Before adding a path, search with
`rg --files` and `rg` and extend the existing owner when one exists.

## Placement

| Content | Location |
| --- | --- |
| Python implementation | `src/slm_training/` or the existing `src/gpu_multi_farm/` package |
| Lean proofs and executable metric checker | `src/leverproof_lean/` |
| Frozen DSL-agnostic harness machinery (versioning, lineage, gate/promotion engines) | `src/slm_training/harness_core/` (see `docs/design/harness-core.md`) |
| DSL analysis helpers (arity, signatures, canonicalization) | `src/slm_training/dsl/analysis/` |
| Runnable entrypoints and maintenance checks | `scripts/` |
| Tests mirroring implementation domains | `tests/` |
| Small committed inputs and expected artifacts | `src/slm_training/resources/` |
| Mirrored external pytest cases (not shipped) | `src/slm_training/resources/test_cases/<test path>.json` |
| Shipped eval and gate policy resources | `src/slm_training/resources/evals/` |
| Git-published immutable model data | `src/slm_training/resources/data/<kind>/<id>/` |
| Human-authored design, operations, and measured evidence | `docs/` |
| OpenWiki-generated agent navigation | `docs/openwiki/` |
| Self-contained Node/frontend packages | `src/apps/` |
| Canonical agent skills | `.agents/skills/` |
| Client discovery links and hooks | `.claude/`, `.cursor/`, `.codex/`, `.github/hooks/` |
| Grok Build project workflows (Rhai orchestration) | `.grok/workflows/` |

The repository root is an allowlist for required manifests and cross-agent
instructions. Application code and owned resources belong below `src/`; generated
documentation belongs below `docs/`. Do not add a new root path without
updating this guide and `scripts/repo_policy.py` in the same reviewed change.
Deployment manifests (`vercel.json` and `.vercelignore`) stay at the root
because Vercel discovers them there.

Ignored model inputs use `outputs/data/<kind>/<id>/`; raw correlated traces and
logs use `outputs/traces/<trace-id>/`. Do not create new sibling data or trace
roots. Use `slm_training.data.store.DataStore` and `slm-data` for resolution,
publication, verification, and legacy migration.

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

## External test and eval resources

Large JSON-shaped pytest tables mirror their test module below
`src/slm_training/resources/test_cases/`. Agents edit inputs directly and use
`python -m scripts.refresh_test_cases <test-or-resource>` for deterministic
snapshot updates; ordinary pytest and CI never rewrite them. The extractor
check (`python -m scripts.extract_test_cases`) prevents large eligible tables
from drifting back inline. These test-only resources are excluded from wheels
and Vercel uploads.

Runtime loss suites and ship-gate policy live under
`src/slm_training/resources/evals/`, remain wheel data, and follow component
versioning. Frozen eval resources are replaced with a new versioned filename,
never edited in place. Ship-gate changes may only preserve or tighten the
committed policy.

## Enforcement

Run the repository policy directly or through the existing changed-file check:

```bash
python -m scripts.repo_policy
.githooks/check-changed
```

The policy rejects unapproved root paths, copied skill mirrors, canonical skills
with no discovery symlink, redundant Codex skill copies, and newly tracked
ignored artifacts. The tracked pre-commit hook and CI run the same check.

Agent hooks add per-edit feedback and are certified identical across harnesses
by `python -m scripts.verify_agent_surfaces`:

- `PreToolUse` rejects raw moves of tracked paths (Claude Code, Codex,
  Copilot CLI).
- `PostToolUse` runs `validate_page_dsl.py --changed`,
  `verify_version_stamps --post-tool-use`, and
  `refresh_test_cases --check --changed` (same three harnesses).
- Cursor and Gemini CLI have no hook mechanism configured; agents there run
  `python -m scripts.repo_policy` and `.githooks/check-changed` by hand.

CI remains authoritative either way. Instruction-file parity across harnesses
is covered in
[`design/agent-harness-parity-audit.md`](design/agent-harness-parity-audit.md).
