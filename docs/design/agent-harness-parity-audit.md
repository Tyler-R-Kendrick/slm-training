# Agent / harness parity audit

Audit of (a) places where the implementation does not match the stated intent
and (b) divergence between the configured coding harnesses. Every claim below
was reproduced against `42d76b2` on 2026-07-25; the reproduction command is
given with each finding.

Canonical law under audit: [`AGENTS.md`](../../AGENTS.md) and
[`decode-invariants.md`](decode-invariants.md).

## Status

**All four phases are implemented.** Findings are kept with their reproduction
commands so the regressions stay recognisable; the fix that closed each one is
recorded inline and in [Fix plan](#fix-plan). One further defect (A3) was found
while verifying Phase 1 and is **not** fixed here — see its entry.

| Guard | Covers |
| --- | --- |
| `python -m scripts.verify_agent_surfaces` | every repository law on every instruction surface, plus hook parity (B1–B3) |
| `python -m scripts.refresh_test_cases --check` | external case schema, stable identities, frozen eval versions, and non-weakened gate policy |
| `tests/test_dsl/test_tokenizer_grammar_invariants.py` | tokenizer layout, pinned without the Node bridge (A1, A2) |
| `python -m scripts.verify_tokenizer_grammar_invariants` | full certificate incl. live-library agreement (A1) |
| `scripts/repo_policy.py::validate_skill_mirrors` | skill mirrors in **both** directions (B6) |

Both new checks run in CI's `python-static` job and, for changed surfaces, in
`.githooks/check-changed`.

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

**Fixed.** `models/dsl_tokenizer.py` now treats
`dsl.openui_tokens.STRUCTURAL_TOKENS` as authoritative for the default DSL
instead of re-deriving it from whichever backend is live; the backend seam
survives only for a non-default `SLM_GRAMMAR_DSL`, where it fails closed.
`lark_backend.structural_tokens()` no longer scrapes capitalised words out of
the grammar text. `vocab_size` is now 569 with the bridge, without the bridge,
and with `node_modules` deleted. `model.twotower` bumped to v246 — the layout
is unchanged; the bump records that it is now pinned.

Why this was an intent violation, not just a bug:

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
- No test pinned `vocab_size == 569`. There is now one, plus a regression test
  asserting the Lark fallback genuinely disagrees (which is why it must not be
  consulted) and one asserting no grammar rule names leak into the token set.

### A2 — The tokenizer certificate could only ever run in the node-enabled CI job

`verify_tokenizer_grammar_invariants` runs only in `ci.yml`'s `python` job,
which does `npm --prefix src/apps/openui_bridge ci` first. The sibling
certificates (`repo_policy`, `verify_decode_invariants`,
`verify_version_stamps`, `verify_checkpoint_references`) all live in
`python-static`, which installs no Node. So the one check that would catch A1 was
structurally incapable of observing the fallback path, and CI was green on
`main` with the drift latent.

**Fixed.** The checkpoint-bound half of the contract — version, `vocab_size`,
and layout SHA for both codecs — is now pinned by plain `pytest` with no Node
required, and `check_changed` selects that file whenever anything under
`src/slm_training/models/` changes. The full certificate still needs the bridge
(it also verifies live-library agreement), so it skips locally with an
actionable reason instead of failing with a misleading one; CI runs the script
directly in the bridge-enabled job, where it cannot be skipped.

#### A2b — a stale vocabulary cap had been red on `main`, unselected

`tests/test_harnesses/model_build/test_dsl_tokenizer.py::test_vocab_is_fixed_and_typed`
asserted `tok.vocab_size <= 512`. Main #920 folded `STRUCTURAL_ID_ATOMS` into
the fixed literal set (505 → 569) and left the cap behind, so the test had been
failing on `main` — with *and* without the Node bridge — while CI stayed green,
because `check_changed --changed-tests-only` runs only changed **test** files
and that file had not changed.

**Fixed** as part of Phase 1: the assertion now reads the checked-in
`tokenizer_layout_registry.json` instead of a hand-written magic number, so it
cannot go stale silently again. Same root cause as A1 — the vocabulary moved
and its guards did not.

### A3 — `pytest` rewrites committed `docs/design/` evidence, nondeterministically (found, NOT fixed)

Found while verifying Phase 1. Running the test suite dirties the repository's
durable experiment ledger:

```bash
git status --short docs/design            # clean
pytest tests/test_scripts/test_run_slm157_flow_consistency_fixture.py
git status --short docs/design            # M iter-slm157-flow-consistency-20260720.{json,md}
```

The test is well-behaved — it passes `--output-dir tmp_path`. The CLI is not:
`scripts/run_slm157_flow_consistency_fixture.py` mirrors into
`docs/design/iter-slm157-flow-consistency-20260720.{json,md}` under a bare
`if args.mode == "fixture":`, ignoring that an explicit `--output-dir` was
given — two lines below it even branches on `args.output_dir is not None` for
the recorded command line.

Two consequences:

- The committed numbers change on every run. Re-running the fixture twice from
  a clean tree produces different values for `E_discrete_flow_rate`,
  `F_random_path_control`, and `G_ar_x22_hybrid_placeholder`, so the
  headline table in a committed measured-results doc is **not reproducible** —
  the opposite of what the iron law says `docs/design/` is for.
- A full `pytest` run also creates ~20 *new* dated `iter-*.json` / `.md`
  records from sibling fixtures with the same pattern, which an agent running
  `git add -A` would commit as if they were real experiment evidence.

**Not fixed here.** 54 of the 76 `scripts/run_*.py` runners share the
mirror-plus-`--output-dir` shape, so this is a systemic change to the
experiment-runner contract rather than a parity fix, and picking the right
semantics (mirror only when `--output-dir` is absent? require an explicit
`--publish`? seed the RNG?) is the experiment owners' call. Suggested minimal
fix: gate the `docs/design/` mirror on `args.output_dir is None`, which makes
the CLI honour the flag it already parses and leaves canonical runs unchanged.
Whether the underlying nondeterminism is intended sampling is a separate
question that a seed would answer.



## B. Cross-harness parity

### B1 — Only the decode invariants have a cross-surface parity check

`verify_decode_invariants.check_agent_surfaces()` enforces that `AGENTS.md`,
`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md` and
`.cursor/rules/decode-invariants.mdc` each carry the decode law. That check
worked — every surface carried it. **No other repository law had an equivalent
check, and every one of them had drifted.** The table below is the state at
audit time; every ❌ is now ✅ and certified.

**Fixed.** `scripts/verify_agent_surfaces.py` owns a declarative
obligation × surface matrix covering thirteen laws plus hook parity;
`verify_decode_invariants.check_agent_surfaces()` delegates to it rather than
keeping a second copy. Missing laws were backfilled into `CLAUDE.md`,
`GEMINI.md`, `.github/copilot-instructions.md`, a new
`.cursor/rules/repo-laws.mdc` (`alwaysApply: true`), and the Grok workflow
header. Surfaces now cite the canonical `I*` ids (B7), and I7 is a documented
invariant section rather than a phantom id.

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
| Agent-owned external cases (`refresh_test_cases`) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Dashboard OpenUI parity | ✅ | ✅ | ❌ | ❌ | ❌ |
| Preregistered campaign law | ✅ | ❌ | ❌ | ❌ | ❌ |
| Serena preference | ✅ | ✅ | ❌ | ❌ | ❌ |
| Checkpoint bucket sync | ✅ | ❌ | ❌ | ❌ | ❌ |
| OpenWiki (don't hand-edit) | ✅ | ✅ | ❌ | ❌ | ❌ |

¹ Copilot's only `versions.json` mention is incidental, inside the decode-invariant
text (`decode.invariants` bump). The general version-stamp law is absent.

Net effect at audit time: a Gemini, Copilot, or Cursor agent could complete a
training run and never be told to update `docs/design/`, the model card, or a
component version — all three of which `AGENTS.md` calls non-optional.

### B2 — Grok is a recognized harness with no instruction surface

`docs/repository-organization.md` lists `.grok/workflows/` as a first-class
location and `.grok` is in `repo_policy.ALLOWED_ROOTS`, but "Grok" appears
**nowhere** in `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`,
`.github/copilot-instructions.md`, or `.cursor/rules/`. `AGENTS.md:3` enumerates
"Cursor, Claude Code, Codex, Gemini, Copilot / GHCP" and omits it.
`.grok/workflows/autotrain.rhai` restates the contracts in its own header
comment — a fourth independent copy of the law with no parity check.

**Fixed.** `.grok/workflows/autotrain.rhai` now names `AGENTS.md` and
`docs/design/decode-invariants.md` in its required-reading and contracts
header, Grok is listed in `AGENTS.md`'s agent line, and the workflow is a
certified surface in the parity matrix for the laws an orchestration script can
act on (canonical AGENTS.md, decode invariants, run cap, iron law, model card).
The `autotrain.continuous-harness-loop` obligation additionally keeps the shared
skill and Grok workflow aligned on continuous default behavior, canonical
`improve-openui-harnesses` and `improve-lean-optimums` routing, merge-from-
`origin/main` provenance, and the between-run `--matrix` view.

### B3 — README overstates hook coverage; the Copilot hook is an empty stub

`README.md:379` claims *"Claude Code, Codex, and Copilot CLI hooks run the same
changed-file checker automatically and reject raw `mv` for tracked paths."*

| Harness | raw-`mv` block | changed-file checker | dashboard parity | version stamp | case resources |
| --- | :-: | :-: | :-: | :-: | :-: |
| Claude Code (`.claude/settings.json`) | ✅ | ❌ | ✅ | ✅ | ✅ |
| Codex (`.codex/hooks.json`) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Copilot (`.github/hooks/`) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Cursor | ❌ (no hook config) | ❌ | ❌ | ❌ | ❌ |

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

**Fixed.** `.github/hooks/changed-tests.json` now carries the post-edit
dashboard-parity, version-stamp, and external-case checks, and the same trio was
mirrored into `.codex/hooks.json`, so all three hook-capable harnesses run the
identical set.
Hook parity is itself certified — `hooks.raw-mv-guard` and
`hooks.post-edit-checks` are obligations in the matrix, so a one-sided hook edit
fails CI. A Claude Code `SessionStart` hook arms `core.hooksPath .githooks` when
it is unset. `README.md` no longer claims the agent hooks run the changed-file
checker; it carries a per-harness coverage table instead, and Cursor's and
Gemini's lack of a hook mechanism is stated in their own instruction surfaces.

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

**Fixed (documented, deliberately not equalised).** `AGENTS.md` now carries an
"MCP servers are not uniform across harnesses" table naming which server each
client has and why: Linear is Claude-only because `autoresearch` issue filing
runs there; Playwright and the HF Hub server are Cursor-only because both have
first-class CLIs (`npx playwright`, `hf`) that every other harness uses instead.
Codex's Serena block moved from a manual-copy `.example` into the committed
`.codex/config.toml`, with the global-copy path kept as a fallback note for
older builds. The divergence was never the defect — being undocumented was.

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

**Fixed.** The refresh block is deleted from `AGENTS.md`, which now points at
`.agents/skills/README.md` as the single owner. The surviving copy drops
`-a codex`, and its normalisation loop covers every canonical skill (not a
hand-maintained list), removes any `.codex/skills/` tree, and ends in
`python -m scripts.repo_policy` so following the documented procedure leaves the
repo green. The `hf skills add --dest=` copy path points at the same loop.

### B6 — `validate_skill_mirrors` only checks one direction

It walks `.claude/skills/` and `.cursor/skills/` looking for orphans and copies,
but never asserts that every `.agents/skills/<name>` **has** a mirror. A newly
added canonical skill stays invisible to Claude Code and Cursor with a green
policy check. Latent today (54/54 mirrored), but nothing prevents it.

**Fixed.** `validate_skill_mirrors` now walks canonical → discovery as well,
reporting `unmirrored skill: <root>/<name> is missing`, with test coverage for
both directions.

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

**Fixed.** All five agent surfaces now cite the canonical `I*` ids, with an
explicit instruction not to renumber locally. `I7` is a real section in
`decode-invariants.md` (it has an enforcing script now), and the phantom `I8`
citations in `verify_decode_invariants` were folded into `I15`, which is where
the doc already covered them. `decode.invariants` bumped to v3.

## Fix plan

All four phases shipped. Deviations from the plan as first written are called
out, since the plan was drafted before the fixes were attempted.

### Phase 1 — Close the tokenizer fallback (A1, A2) — done

1. **Backend seam no longer decides the vocabulary.** The plan proposed
   comparing the backend-derived set against the constant and raising on
   mismatch. That would have made every torch-free import of
   `models.dsl_tokenizer` require Node. Implemented instead: for the default
   `openui` DSL the `dsl.openui_tokens` constant *is* authoritative, so the
   frozen contract is deterministic by construction. The backend seam survives
   only for a non-default `SLM_GRAMMAR_DSL`, where it fails closed rather than
   handing back the OpenUI vocabulary for a different grammar. The
   backend-agreement assertion moved to the certificate, which is the right
   place for a check that legitimately needs the live library.
2. **Lark no longer leaks grammar internals.** The leak was in
   `structural_tokens()`, not `component_names()` as the plan assumed — a
   `\b([A-Z][A-Za-z0-9]+)\b` scrape of the raw grammar text. It now extracts
   quoted literals, which is what a surface token actually is.
3. **Layout pinned by test** for both codecs (version, `vocab_size`, layout
   SHA), with no Node required. `check_changed` selects the file for any change
   under `src/slm_training/models/`.
4. **Assertion text fixed.** A missing bridge now says so and names
   `npm --prefix src/apps/openui_bridge ci`; a genuine layout change says the
   registry records what checkpoints were trained against, so an accidental
   layout must not be written into it.
5. **Certificate left in the node-enabled job**, rather than adding ~30s of
   `npm ci` to `python-static` for no new coverage: the bridge-free pins now
   carry the checkpoint-bound half, and the certificate skips locally with an
   actionable reason while CI runs the script directly where it cannot skip.
6. `model.twotower` v245 → v246.

A3 was found during this phase's verification and is deliberately left open.

### Phase 2 — One enforcement mechanism for cross-surface parity (B1, B2, B6, B7) — done

1. `scripts/verify_agent_surfaces.py` holds the declarative
   obligation × surface matrix — thirteen instruction laws plus two hook laws.
2. `verify_decode_invariants.check_agent_surfaces()` delegates to it.
3. Wired into `ci.yml`'s `python-static` job and into `check_changed.check()`,
   which re-certifies whenever any surface file changes.
4. Backfilled `CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`, and
   a new `.cursor/rules/repo-laws.mdc` (`alwaysApply: true`). Markers are skill
   and script names, so surfaces stay pointers rather than a third copy of the
   prose.
5. `repo_policy.validate_skill_mirrors` gained the reverse direction.
6. Surfaces renumbered to the canonical `I*` ids; `I7` promoted from a phantom
   id cited only in Python docstrings to a documented section; `I8` folded into
   `I15` where the doc already covered it. `decode.invariants` v2 → v3.

### Phase 3 — Level the hooks (B3) — done

Both halves, not either/or: the hooks were implemented **and** the README claim
corrected. `.github/hooks/changed-tests.json` and `.codex/hooks.json` now carry
the same `PostToolUse` trio as `.claude/settings.json`; hook parity is itself
certified by the `hooks.raw-mv-guard` and `hooks.post-edit-checks` obligations,
so a one-sided hook edit fails CI. A Claude Code `SessionStart` hook arms
`core.hooksPath .githooks` when unset. `README.md` carries a per-harness
coverage table, and Cursor's and Gemini's lack of a hook mechanism is stated in
their own instruction surfaces rather than only in `AGENTS.md`.

### Phase 4 — De-duplicate and correct the install instructions (B4, B5) — done

1. Refresh block deleted from `AGENTS.md`; `.agents/skills/README.md` is the
   single owner.
2. The surviving copy drops `-a codex`, and its normalisation loop iterates
   every canonical skill rather than a hand-maintained list, removes any
   `.codex/skills/` tree, and ends in `python -m scripts.repo_policy`.
3. MCP divergence documented rather than equalised — each server is where it is
   for a reason, and the defect was that the reason was unwritten.
4. Codex's Serena block promoted into the committed `.codex/config.toml`, with
   the `~/.codex/config.toml` copy kept as a fallback note for older builds.

### Open — A3

Not part of the four phases; see [A3](#a3--pytest-rewrites-committed-docsdesign-evidence-nondeterministically-found-not-fixed).
Suggested minimal fix is recorded there.

## Verification

```bash
python -m scripts.repo_policy
python -m scripts.verify_agent_surfaces
python -m scripts.verify_decode_invariants
python -m scripts.verify_version_stamps --check
python -m scripts.verify_checkpoint_references --check
python scripts/validate_page_dsl.py --check
pytest tests/test_scripts tests/test_dsl tests/test_models tests/test_harnesses/model_build

# Needs the Node bridge; that is the point of A1, and it now says so when absent
npm --prefix src/apps/openui_bridge ci
python -m scripts.verify_tokenizer_grammar_invariants
```

The A1 regression, reproducible on any clone:

```bash
mv src/apps/openui_bridge/node_modules /tmp/nm
python -c "from slm_training.models.dsl_tokenizer import DSLNativeTokenizer as T; print(T.build().vocab_size)"
mv /tmp/nm src/apps/openui_bridge/node_modules
# was: 605 without the bridge, 569 with it. now: 569 either way.
```

After running the suite, check `git status --short docs/design` — A3 means a
plain `pytest` run leaves the evidence ledger dirty.
