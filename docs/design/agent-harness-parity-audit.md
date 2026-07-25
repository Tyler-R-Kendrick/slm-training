# Agent / harness parity audit

Audit of (a) places where the implementation does not match the stated intent
and (b) divergence between the configured coding harnesses. Every claim below
was reproduced against `42d76b2` on 2026-07-25; the reproduction command is
given with each finding.

Canonical law under audit: [`AGENTS.md`](../../AGENTS.md) and
[`decode-invariants.md`](decode-invariants.md).

## Method

- Ran every repository certificate (`repo_policy`, `verify_decode_invariants`,
  `verify_version_stamps`, `verify_checkpoint_references`,
  `verify_tokenizer_grammar_invariants`, `validate_page_dsl`) in a clean 3.12
  virtualenv, with and without `src/apps/openui_bridge/node_modules` present.
- Diffed each agent surface (`AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
  `.github/copilot-instructions.md`, `.cursor/rules/*.mdc`, `.codex/`,
  `.grok/`) against the obligation list in `AGENTS.md`.
- Diffed the hook and MCP configuration of every harness.

## A. Implementation vs. intent

### A1 — The output tokenizer vocabulary changes size with the Node bridge (critical)

`DSLNativeTokenizer.build()` produces **569** tokens when
`src/apps/openui_bridge/node_modules` is installed and **605** when it is not.
`DSL_TOKENIZER_VERSION` stays `5` in both cases and
`resources/tokenizer_layout_registry.json` pins `569`, so the divergence is
silent at the checkpoint boundary.

```bash
python -c "from slm_training.models.dsl_tokenizer import DSLNativeTokenizer as T; print(T.build().vocab_size)"
# 569 with src/apps/openui_bridge/node_modules present, 605 without
```

Mechanism:

1. `models/dsl_tokenizer.py:36` computes `STRUCTURAL_TOKENS` **at import time**
   via `get_backend("openui").structural_tokens()`, under a bare
   `except Exception` the comment itself labels *"fail-open fallback"*.
2. `dsl/grammar/backends/openui_hybrid.py` resolves `_active()` to the
   lang-core backend when the Node bridge is importable, otherwise to the
   **Lark** backend.
3. `lark_backend.component_names()` returns the grammar's rule and terminal
   names. `_COMPONENT_NAMES` therefore grows 35 → 71 and admits
   `AST`, `BOOL`, `COMMENT`, `NAME`, `NUMBER`, `STRING`, `NULL`, `Lang`,
   `OpenUI`, `Component`, `Priority`, and `Za` — the last being a fragment of
   the `[A-Za-z0-9_]` character class — into a **model vocabulary**.

Why this is an intent violation, not just a bug:

- The docstring at `models/dsl_tokenizer.py:38-43` asserts the fallback is
  *"identical for the default OpenUI backend, so vocab layout is unchanged"*.
  It is not.
- `AGENTS.md` § I.2 requires production paths to **fail closed** and forbids
  widening a constrained domain on failure. This widens the vocabulary itself.
- Two checkpoints trained on either side of this boundary have different
  embedding widths, and `DSLNativeTokenizer` load-time guards only compare
  `version`, which matches.
- `scripts/verify_tokenizer_grammar_invariants.py:71` is the guard, and its
  failure text — *"bump its tokenizer version and update the checked-in layout
  registry with a migration note"* — **instructs the wrong fix**. An agent that
  runs it without `node_modules` is told to write `605` into the registry,
  which would commit the corruption.
- `tests/test_dsl/test_tokenizer_grammar_invariants.py::test_certified_tokenizer_grammar_invariants`
  fails locally for the same reason, with the same misleading message.
- No test pins `vocab_size == 569`.

### A2 — The tokenizer certificate can only ever run in the node-enabled CI job

`verify_tokenizer_grammar_invariants` runs only in `ci.yml`'s `python` job,
which does `npm --prefix src/apps/openui_bridge ci` first. The sibling
certificates (`repo_policy`, `verify_decode_invariants`,
`verify_version_stamps`, `verify_checkpoint_references`) all live in
`python-static`, which installs no Node. So the one check that would catch A1
is structurally incapable of observing the fallback path, and CI is green on
`main` today with the drift latent.

## B. Cross-harness parity

### B1 — Only the decode invariants have a cross-surface parity check

`verify_decode_invariants.check_agent_surfaces()` enforces that `AGENTS.md`,
`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md` and
`.cursor/rules/decode-invariants.mdc` each carry the decode law. That check
works — every surface carries it. **No other repository law has an equivalent
check, and every one of them has drifted.**

| Obligation (AGENTS.md) | AGENTS | CLAUDE | GEMINI | Copilot | Cursor rules |
| --- | :-: | :-: | :-: | :-: | :-: |
| Decode invariants | ✅ | ✅ | ✅ | ✅ | ✅ |
| Hard run cap (`MAX_RUN_MINUTES`) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Iron law: `documenting-experiment-results` | ✅ | ✅ | ✅ | ❌ | ❌ |
| `honest-ship-eval` / never weaken gates | ✅ | ❌ | ❌ | ❌ | ❌ |
| Data-quality law: `synthesis-feedback` | ✅ | ✅ | ❌ | ❌ | ❌ |
| `organize-repository` + `git mv` | ✅ | ✅ | ✅ | ✅ | ✅ |
| Model card + README summary | ✅ | ✅ | ✅ | ❌ | ❌ |
| Version stamps / component bump | ✅ | ✅ | ❌ | ⚠️¹ | ❌ |
| Dashboard OpenUI parity | ✅ | ✅ | ❌ | ❌ | ❌ |
| Preregistered campaign law | ✅ | ❌ | ❌ | ❌ | ❌ |
| Serena preference | ✅ | ✅ | ❌ | ❌ | ❌ |
| Checkpoint bucket sync | ✅ | ❌ | ❌ | ❌ | ❌ |
| OpenWiki (don't hand-edit) | ✅ | ✅ | ❌ | ❌ | ❌ |

¹ Copilot's only `versions.json` mention is incidental, inside the decode-invariant
text (`decode.invariants` bump). The general version-stamp law is absent.

Net effect: a Gemini, Copilot, or Cursor agent can complete a training run and
never be told to update `docs/design/`, the model card, or a component version —
all three of which `AGENTS.md` calls non-optional.

### B2 — Grok is a recognized harness with no instruction surface

`docs/repository-organization.md` lists `.grok/workflows/` as a first-class
location and `.grok` is in `repo_policy.ALLOWED_ROOTS`, but "Grok" appears
**nowhere** in `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
`.github/copilot-instructions.md`, or `.cursor/rules/`. `AGENTS.md:3` enumerates
"Cursor, Claude Code, Codex, Gemini, Copilot / GHCP" and omits it.
`.grok/workflows/autotrain.rhai` restates the contracts in its own header
comment — a fourth independent copy of the law with no parity check.

### B3 — README overstates hook coverage; the Copilot hook is an empty stub

`README.md:379` claims *"Claude Code, Codex, and Copilot CLI hooks run the same
changed-file checker automatically and reject raw `mv` for tracked paths."*

| Harness | raw-`mv` block | changed-file checker | dashboard parity | version stamp |
| --- | :-: | :-: | :-: | :-: |
| Claude Code (`.claude/settings.json`) | ✅ | ❌ | ✅ | ✅ |
| Codex (`.codex/hooks.json`) | ✅ | ❌ | ❌ | ❌ |
| Copilot (`.github/hooks/`) | ✅ | ❌ | ❌ | ❌ |
| Cursor | ❌ (no hook config) | ❌ | ❌ | ❌ |

The changed-file checker runs in **none** of them.
`.github/hooks/changed-tests.json` — the file named for it — is an empty stub:

```json
{ "version": 1, "hooks": {} }
```

Only `.githooks/pre-commit` runs it, and that requires
`git config core.hooksPath .githooks` once per clone (not set in a fresh CCR
checkout). Separately, `AGENTS.md` § Normalized component versioning claims the
bump rule is enforced by "CI, pre-commit, **agent hooks**" — true only for
Claude Code.

### B4 — MCP server sets diverge across harnesses

| Config | Serena | Playwright | Hugging Face | Linear |
| --- | :-: | :-: | :-: | :-: |
| `.mcp.json` (Claude Code) | ✅ | ❌ | ❌ | ✅ |
| `.cursor/mcp.json` | ✅ | ✅ | ✅ | ❌ |
| `.vscode/mcp.json` (Copilot Chat) | ✅ | ❌ | ❌ | ❌ |
| `.codex/serena.config.toml.example` | ⚠️ manual copy | ❌ | ❌ | ❌ |

The `playwright-cli` skill is advertised to every agent but only Cursor has the
server; the `autoresearch` skill's "file ideas as Linear issues" step only works
under Claude Code. `AGENTS.md`'s client table presents Codex's Serena as
configured, but `.codex/config.toml` only sets `[features] hooks = true` — the
Serena block is an `.example` requiring a manual copy to `~/.codex/config.toml`.

### B5 — The documented skill-refresh commands produce a state `repo_policy` rejects

`AGENTS.md:206-216` and `.agents/skills/README.md` both tell agents to run:

```bash
npx skills add DietrichGebert/ponytail --skill '*' \
  -a claude-code -a cursor -a codex -a github-copilot -y --copy
```

`--copy` writes real directories into `.claude/skills/` and `.cursor/skills/`,
which `repo_policy.validate_skill_mirrors` rejects as
`copied skill mirror: …; use a symlink`. `-a codex` writes `.codex/skills/`,
which the same function rejects as `redundant Codex skill copy`. `AGENTS.md`'s
block offers only a parenthetical "then re-symlink … (see skills README)" and no
`.codex/skills` cleanup at all.

The block is also duplicated between `AGENTS.md` and `.agents/skills/README.md`
and has **already diverged** — the README carries the re-symlink loop and the
headroom-helpers note, `AGENTS.md` does not. `organize-repository` forbids
exactly this ("Do not add a second helper, schema, config, or guide for an
existing concern").

### B6 — `validate_skill_mirrors` only checks one direction

It walks `.claude/skills/` and `.cursor/skills/` looking for orphans and copies,
but never asserts that every `.agents/skills/<name>` **has** a mirror. A newly
added canonical skill stays invisible to Claude Code and Cursor with a green
policy check. Latent today (54/54 mirrored), but nothing prevents it.

### B7 — Invariant numbering is inconsistent between the canonical doc and every agent surface

`decode-invariants.md` uses `I1–I6, I9, I10, I10b, I11, I12, I13, I14, I15`
(there are no `### I7`/`### I8` sections; those IDs appear only in
`verify_decode_invariants` docstrings). All five agent surfaces renumber the
same list `1–15` in a **different order**:

| Concept | Canonical doc | Agent surfaces |
| --- | --- | --- |
| Deterministic bypass outranks learned scores | `I1` | `4` |
| Shared encoder↔decoder ops vocabulary | `I13` | `11` |
| Multi-turn CRDT event store | `I11` | `12` |

"Invariant 11" therefore means two different things depending on which file the
reader has open, in a document set whose entire purpose is unambiguous citation.

## Fix plan

Ordered by risk. Each phase is independently shippable.

### Phase 1 — Close the tokenizer fallback (A1, A2)

1. **Make the backend seam fail closed.** Replace the bare `except Exception`
   in `models/dsl_tokenizer.py:_active_structural_tokens()` with either an
   explicit raise, or — preferred, since the tokenizer must build in torch-free
   contexts — a check that the backend-derived set is a superset-free match for
   `openui_tokens.STRUCTURAL_TOKENS`, raising when it is not. The vocabulary is
   a frozen contract; deriving it from a *fallback* backend is the defect.
2. **Stop the Lark backend leaking grammar internals.** `lark_backend
   .component_names()` returning rule/terminal names (`AST`, `NAME`, `Za`) is
   wrong independent of the tokenizer; restrict it to the declared component
   set.
3. **Pin the layout in a test**, not only in the registry:
   `assert DSLNativeTokenizer.build().vocab_size == 569` plus the layout sha,
   so the drift is caught by `pytest` and by `.githooks/check-changed`, not only
   by a node-enabled CI job.
4. **Fix the misleading assertion text** in
   `verify_tokenizer_grammar_invariants.py:71` — when the bridge is absent it
   must say so ("OpenUI bridge unavailable; run `npm --prefix
   src/apps/openui_bridge ci`") rather than inviting a registry bump.
5. **Move the certificate into `python-static`** (adding the bridge install
   there), or fail it loudly when the bridge is missing. Today it cannot
   observe the failure mode it exists to catch.
6. Bump the tokenizer/decode components in `resources/versions.json` per the
   version-stamp contract and record the result in a dated
   `docs/design/` measured-results note.

### Phase 2 — One enforcement mechanism for cross-surface parity (B1, B2, B6, B7)

The `check_agent_surfaces()` pattern already works; the problem is that it
covers exactly one law. Generalize rather than add a second checker:

1. Add `scripts/verify_agent_surfaces.py` holding a **declarative
   obligation × surface matrix** — obligation id, required marker strings, and
   the surfaces that must carry it (including `.grok/workflows/autotrain.rhai`).
2. Have `verify_decode_invariants.check_agent_surfaces()` delegate to it so the
   decode law stays enforced through one owner, not two.
3. Wire it into `ci.yml`'s `python-static` job and into
   `scripts/check_changed.check()` next to the existing `verify_version_stamps`
   call, so pre-commit catches it too.
4. Backfill the missing obligations into `GEMINI.md`,
   `.github/copilot-instructions.md`, and a new
   `.cursor/rules/repo-laws.mdc` (`alwaysApply: true`) until the matrix is
   green. Keep each surface a *pointer* to `AGENTS.md`, not a third copy of the
   prose — the marker strings should be the skill names and script names, which
   is what actually needs to reach the agent.
5. Extend `repo_policy.validate_skill_mirrors` with the reverse direction:
   every `.agents/skills/<name>` must have a symlink under both discovery roots.
6. Renumber the agent-surface invariant lists to the canonical `I*` ids from
   `decode-invariants.md` (or renumber the doc — either way, one scheme), and
   add the id strings to the parity matrix so they cannot drift again.

### Phase 3 — Level the hooks (B3)

1. Either implement `.github/hooks/changed-tests.json` and add equivalent
   `PostToolUse` entries to `.codex/hooks.json`, **or** correct `README.md:379`
   and `AGENTS.md` to describe what the hooks actually do. Do not leave the
   claim and the stub both in place.
2. Mirror Claude Code's `PostToolUse` pair (`validate_page_dsl.py --changed`,
   `verify_version_stamps --post-tool-use`) into `.codex/hooks.json`. These are
   the two laws with automated feedback; restricting them to one harness is the
   single largest behavioural difference between agents.
3. Add a `SessionStart` hook (or document the step) that sets
   `core.hooksPath .githooks`, so the changed-file checker is actually armed in
   a fresh clone.
4. Cursor has no hook surface; state that explicitly in `AGENTS.md` rather than
   implying uniform coverage.

### Phase 4 — De-duplicate and correct the install instructions (B4, B5)

1. Delete the refresh block from `AGENTS.md` and link
   `.agents/skills/README.md` as the single owner (or the reverse) — the two
   copies have already diverged.
2. Fix the surviving copy: drop `-a codex` (Codex reads `.agents/skills/`, and
   `.codex/skills/` is policy-rejected), and fold the re-symlink loop and
   `hf skills add --dest=.cursor/skills` copy-cleanup into the same block, so
   following the documented procedure leaves `repo_policy` green.
3. Decide MCP parity deliberately: either add Playwright + Hugging Face to
   `.mcp.json` and Linear to `.cursor/mcp.json`, or document in `AGENTS.md`
   which skills are harness-limited. Today the divergence is undocumented.
4. Promote `.codex/serena.config.toml.example` to a committed project-local
   `.codex/config.toml` block, or mark it clearly as manual in the client table.

## Verification

```bash
python -m scripts.repo_policy
python -m scripts.verify_decode_invariants
python -m scripts.verify_version_stamps --check
python -m scripts.verify_tokenizer_grammar_invariants   # requires the Node bridge today
npm --prefix src/apps/openui_bridge ci                  # ...which is the point of A1
python scripts/validate_page_dsl.py --check
.githooks/check-changed
```

Status at audit time: all green **except** `verify_tokenizer_grammar_invariants`,
which fails without the Node bridge and passes with it.
