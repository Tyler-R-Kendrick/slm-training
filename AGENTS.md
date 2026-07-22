# Agent instructions (all coding agents)

Applies to **every** coding agent (Cursor, Claude Code, Codex, Gemini, Copilot /
GHCP, others). Prefer this file over tool-specific defaults on process conflicts.

@RTK.md

## Repo goals

Experiment-first OpenUI layout SLMs:

1. **Honest models** — TwoTower / grammar-diffusion that clear multi-suite
   `--ship-gates`, not fixture memorizers (`docs/design/adversarial-review.md`).
2. **Measurable progress** — every train / eval / bench / matrix run leaves
   durable evidence under `docs/design/`.
3. **Research → code → results** — specs cite papers; harnesses implement
   levers; docs record what ran and whether gates passed.
4. **Ship vs demo** — fixture demos are wiring-only; production claims need full
   scoreboards (full `rico_held` / HF / DESIGN.md when claimed).
5. **Durable checkpoints** — real full HF-context trains upload checkpoints to
   the [OpenUI HF Bucket](https://huggingface.co/buckets/TKendrick/OpenUI)
   (`docs/design/checkpoint-bucket.md`).
6. **Model cards** — every new/promoted checkpoint updates
   [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) **and** the README “Model card
   (summary)” section.
7. **Standard evals** — every evaluation run emits AgentEvals JSONL and is
   executed/published with the pinned AgentV SDK; domain metrics and honest ship
   gates remain authoritative (`docs/design/agentv-evaluation.md`).

## Hard run cap

Every train, eval, benchmark, profile, telemetry, matrix, reproduction, and
supporting shell command must obey the canonical cap in
`src/slm_training/levers.py`. Use its derived interrupt and kill-grace values;
training, campaign, and CI harnesses must not exceed its `MAX_RUN_MINUTES`.
Change that one constant, then run
`python -m scripts.repo_policy --sync-run-policy` to regenerate the GitHub and
Vercel adapters. Prefer local compute;
remote CI and managed jobs are last-resort convenience surfaces. A timed
out, interrupted, or killed run is never evidence.

Start: `README.md`, `docs/MODEL_CARD.md`, `docs/design/openui-twotower.md`,
`docs/design/quality-experiment-matrix.md`,
`docs/design/perf-experiment-matrix.md`, `docs/design/research-lineage.md`,
`docs/design/checkpoint-bucket.md`, `docs/repository-organization.md`.

## Skills

Canonical: **`.agents/skills/<name>/SKILL.md`**. Mirrored for discovery under
`.claude/skills/` and `.cursor/skills/` with symlinks. Edit only the canonical
copy; Codex and GitHub Copilot discover `.agents/skills/` directly.

**If a skill might apply (~1%), open and follow it before acting.**

| Skill | Use when |
| --- | --- |
| `documenting-experiment-results` | After any train / eval / bench / profile / matrix / telemetry run |
| `honest-ship-eval` | Eval, gates, readiness claims, metric changes, demo vs ship |
| `running-experiment-matrices` | Running or extending E* / X* / PQR / phase matrices |
| `openui-autoresearch` | Evidence-grounded campaigns, data/researcher repair, telemetry persistence, and RL readiness |
| `improve-openui-harnesses` | Enhancing canonical research, data, model, eval, preference, distill, promotion, annotation, quality, or RL harnesses without parallel paths or artifact sprawl |
| `autotrain` | Running any training pipeline phase (train/test data, SFT, eval, distill, preference, RL, experiments, checkpoints, annotations, bench, autoresearch self-improvement + hypothesis loop) — per-phase references load on demand |
| `autoresearch` | Knowledge-driven research orchestration: read/update repo + personal brains (OpenWiki / OKF / Obsidian), run the prior-work discovery loop, drive the autotrain hypothesis loop, and file ideas/experiments as Linear issues/milestones/projects — per-stage references load on demand |
| `ponytail` (+ `-review` / `-audit` / …) | Any coding task — write the minimum that works (YAGNI ladder) |
| `organize-repository` | Creating, moving, renaming, deleting, or duplicating tracked paths; adding modules/docs/src/apps/skills; repository-sprawl review |
| `caveman` (+ `-commit` / `-review` / …) | Opt-in terse chat / short commits / one-line review comments |
| `headroom` | Large tool outputs, logs, greps, or context pressure |
| `rtk` | Verbose shell output — prefer `rtk <cmd>` when installed ([`RTK.md`](RTK.md)) |
| `hf-cli` | Hub models/datasets/spaces, auth, cache, HF jobs, buckets, downloads |
| `huggingface-*` / `hf-*` / `trl-training` / … | Other [huggingface/skills](https://github.com/huggingface/skills) workflows (papers, datasets viewer, trainers, Spaces, memory estimate, …) |
| `playwright-cli` | Browser automation or playground e2e |
| `synthesis-feedback` | After any training-data build/synthesis: read `quality_report.json` + `rejected.jsonl` + `synthesis_feedback.json`, fix the synthesis harness (never the gates), file the emitted experiment candidates |
| `frontier-describe` | Fill train-only frozen frontier artifacts and validate leakage/coverage |
| `dashboard-openui-parity` | Editing a dashboard page (`src/apps/dashboard/src/pages/*.tsx`) — keep its interpreted-mode `static/openui/*.openui` program at parity |

### Token-efficiency stack (ponytail · caveman · headroom · rtk)

Installed into **`.agents/skills/`** and discovered by Claude Code
(`.claude/skills/`), Cursor / Codex / GitHub Copilot (project `.agents/skills/`),
with Cursor rule files under [`.cursor/rules/`](.cursor/rules/) and GHCP under
[`.github/copilot-instructions.md`](.github/copilot-instructions.md).

| Layer | What it saves | Default |
| --- | --- | --- |
| [ponytail](https://github.com/DietrichGebert/ponytail) | Less code written | Always on for coding (skills + Cursor rules) |
| [caveman](https://github.com/JuliusBrussee/caveman) | Shorter agent prose | Opt-in (`/caveman` or “talk like caveman”) |
| [headroom](https://github.com/roman-ryzenadvanced/headroom-skill) | Smaller pasted tool results | When outputs are large / context is tight |
| [rtk](https://github.com/rtk-ai/rtk) | Smaller shell command output | Prefer `rtk` when binary available |

Refresh / reinstall (project, four agents):

```bash
npx skills add DietrichGebert/ponytail --skill '*' \
  -a claude-code -a cursor -a codex -a github-copilot -y --copy
npx skills add JuliusBrussee/caveman \
  --skill caveman --skill caveman-commit --skill caveman-review \
  --skill caveman-help --skill caveman-compress --skill caveman-stats \
  -a claude-code -a cursor -a codex -a github-copilot -y --copy
npx skills add roman-ryzenadvanced/headroom-skill --skill headroom \
  -a claude-code -a cursor -a codex -a github-copilot -y --copy
# then re-symlink .claude/.cursor discovery → .agents/skills (see skills README)
curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
rtk init --copilot   # project GHCP hooks + instructions
```

Optional **plugin** installs (Claude Code / Codex / Copilot CLI) when you want
host lifecycle hooks beyond skills:

```text
# Claude Code
/plugin marketplace add DietrichGebert/ponytail
/plugin install ponytail@ponytail
/plugin marketplace add JuliusBrussee/caveman
/plugin install caveman@caveman

# Codex CLI
codex plugin marketplace add DietrichGebert/ponytail && codex plugin add ponytail@ponytail

# GitHub Copilot CLI (ghcp)
copilot plugin marketplace add DietrichGebert/ponytail
copilot plugin install ponytail@ponytail
```

Optional full Headroom proxy (heavier than the portable skill):
`uv tool install "headroom-ai[all]"` then `headroom wrap claude|codex|copilot|cursor`.

### Hugging Face skills + CLI

Source: [huggingface/skills](https://github.com/huggingface/skills) (Cursor:
marketplace installs `hf-cli`; additional skills via `hf skills add`).

Project install (already in `.agents/skills/`, symlinked for Cursor/Claude):

```bash
curl -LsSf https://hf.co/cli/install.sh | bash   # once per machine
hf skills add --force                            # regenerate hf-cli
hf skills add <name> --force                     # one marketplace skill
hf skills update                                 # refresh installed marketplace skills
hf skills add --claude --force                   # Claude discovery symlinks
hf skills add --dest=.cursor/skills --force      # Cursor discovery (hf-cli)
```

Cursor also loads MCP from [`.cursor/mcp.json`](.cursor/mcp.json) (Playwright +
Hugging Face Hub MCP + **Serena**). Optional UI install:
[Cursor marketplace — Hugging Face](https://cursor.com/marketplace/huggingface).

### Serena MCP (semantic code navigation)

[Serena](https://github.com/oraios/serena) provides IDE-like symbolic tools
(find symbol / references / rename / replace body) via MCP. Do **not** install
from MCP marketplaces — use the official quick start:

```bash
# prerequisite: uv (https://docs.astral.sh/uv/)
uv tool install -p 3.13 serena-agent
serena init
cd /path/to/slm-training
serena project create --language python --language typescript --index
# health: serena project health-check
```

Project config: [`.serena/project.yml`](.serena/project.yml) (committed). Cache /
local overrides stay gitignored under `.serena/`.

| Client | Config in this repo |
| --- | --- |
| Cursor | [`.cursor/mcp.json`](.cursor/mcp.json) (`--context ide --project .`) |
| Claude Code | [`.mcp.json`](.mcp.json) + hooks in [`.claude/settings.json`](.claude/settings.json) |
| VS Code / Copilot Chat | [`.vscode/mcp.json`](.vscode/mcp.json) |
| Codex | copy [`.codex/serena.config.toml.example`](.codex/serena.config.toml.example) → `~/.codex/config.toml` (or `serena setup codex`) |
| Copilot CLI | `/mcp add` → `serena start-mcp-server --context=copilot-cli --project-from-cwd` |

Prefer Serena symbolic tools over raw grep/read when navigating `src/` /
`scripts/`. Docs: https://oraios.github.io/serena/

Prefer `hf` over deprecated `huggingface-cli`. Auth: `hf auth login` /
`hf auth whoami`. CLI docs:
https://huggingface.co/docs/huggingface_hub/guides/cli

### Checkpoint bucket (full training runs)

**Bucket:** `hf://buckets/TKendrick/OpenUI` →
https://huggingface.co/buckets/TKendrick/OpenUI

| Run kind | Checkpoints |
| --- | --- |
| Full HF-context train (`train_model` / `hf_jobs_train` / `remote_train`) | Sync to bucket under `checkpoints/<run_id>/` |
| Scratch matrix / CI / fixture demo | Local `outputs/` only (`--no-sync-checkpoints`) |

**GPU host:** Prefer [HF Jobs](docs/design/hf-jobs-train.md)
(`python -m scripts.hf_jobs_train --dry-run`) or pods (`remote_train`). Do **not**
use Spaces ZeroGPU for full trains (short quotas, no `torch.compile`).

```bash
export HF_TOKEN=hf_...   # required for write; never commit
python -m scripts.train_model --train-dir outputs/data/train/v1 \
  --context-backend hf --run-id twotower_v1 --steps 200 --fast-train
# Managed GPU Job (A10G+):
python -m scripts.hf_jobs_train --run-id twotower_v1 --steps 200 --branch main
# Manual / rescue sync:
python -m scripts.sync_checkpoints --run-dir outputs/runs/twotower_v1 --ensure-bucket
```

Agents must **not** treat a full HF train as done until
`train_summary.json` contains `checkpoint_bucket` with a successful remote URI
(or an explicit documented `--no-sync-checkpoints` / scratch reason). Use
`hf-cli` / bucket skills for inspection (`hf buckets list TKendrick/OpenUI -R`).

### Model card (required with every checkpoint)

Whenever a checkpoint is **created, synced, bootstrapped, or promoted**:

1. Update **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)** — roster row, eval table
   (suite `n` + metrics + pass/fail), recipe (device/steps/backend/honesty),
   bucket URI or local path, and append **Checkpoint history**.
2. Refresh **README → “Model card (summary)”** — short table only; link to the
   full card for detail. Do not let the README diverge from the card.
3. Keep claims honest (fixture / scratch matrix ≠ production HF ship).

Triggers include: `train_model`, `hf_jobs_train`, `remote_train`, `bootstrap_playground`,
`sync_checkpoints`, matrix runs that designate a reusable champion, preference /
RL stages that write a new serving `*.pt`, and `--register-promoted`.

A checkpoint without a model-card + README summary update is incomplete work
(same bar as missing `docs/design/` measured-results).

### Dashboard OpenUI parity (keep DSL in sync)

The dashboard renders every page **two ways**, switchable by the sidebar
**◈ Compiled / ◇ Interpreted** toggle: hand-written React
(`src/apps/dashboard/src/pages/*.tsx`, *compiled*) and a committed **OpenUI Lang**
program (`src/slm_training/web/static/openui/<slug>.openui`, *interpreted*) run live
through the official `@openuidev` `<Renderer>` with the dashboard's hybrid library +
`/api` tool provider (`src/apps/dashboard/src/interpret/`). **They must stay at parity.**

Whenever you change a page (`pages/*.tsx`), its shared components
(`components.tsx`), or add/remove a route (`main.tsx`): update the matching `.openui`
program (and any `library.tsx` component / `toolProvider.ts` query it needs), then run
`python scripts/validate_page_dsl.py` (rewrites `static/openui/MANIFEST.json`). A page
change that leaves interpreted mode wrong is incomplete work. Full loop + gotchas:
**REQUIRED SKILL:** `dashboard-openui-parity`. These programs use the full OpenUI Lang
(not the placeholder training subset) — validate with `validate_page_dsl.py`, never the
training bridge; training data + ship-gates are untouched.

## Iron law: docs follow every experiment

```text
NO TRAIN / EVAL / BENCH / PROFILE / TELEMETRY / MATRIX / REPRO
WITHOUT UPDATING DOCS
```

Numbers only in `outputs/`, chat, or a PR comment = incomplete work.

**Triggers (complete, whether invoked directly or via the `slm` wrapper):**
`train_model`, `train_rl`, `train_preference`,
`remote_train`, `hf_jobs_train`, `evaluate_model`, `evaluate_loss_suites`, `diagnose_eval`,
`run_quality_matrix`, `run_grammar_matrix`, `run_perf_matrix`,
`run_phase_pipeline`, `reproduce_baseline`, `run_scaling_ladder`,
`run_mixture_search`, `bench_*` (incl. telemetry/accel), `profile_generate`,
or any ad-hoc run whose scoreboard / gates / latency inform a decision.

**Required each time:**

1. AgentEvals JSONL plus an AgentV SDK result bundle for every eval run. New
   eval entrypoints use `src/slm_training/evals/agentv.py`; no alternate run
   envelope.
2. JSON under `docs/design/` (scripts often mirror; verify it matches this run).
3. Matching markdown measured-results / notes updated (not JSON-only).
4. Recipe metadata: device, steps, backend, matrix set, suite `n`, honesty mode.
5. Honest pass/fail vs `--ship-gates` or perf guardrails.
6. If a checkpoint was written/promoted: update `docs/MODEL_CARD.md` **and**
   README “Model card (summary)”.
7. Commit docs with the experiment — no “docs later” TODO.
8. Result JSON carries a `version_stamp` (schema `version_stamp/v1`; canonical
   writers emit it). If you changed any metric, gate, harness, matrix, or
   data-builder file watched by `src/slm_training/resources/versions.json`,
   bump that component (or append a `no-bump: <reason>` history note) in the
   same change — `python -m scripts.verify_version_stamps --check` enforces it.

**Doc homes:** quality/ship → `quality-experiment-matrix.md` (+ adversarial
review on policy changes); perf → `perf-experiment-matrix.md` /
`runtime-performance.md`; checkpoints → `MODEL_CARD.md` + README summary +
`checkpoint-bucket.md`; lever-specific → that design doc.

## Data-quality law: every synthesis closes its own loop

```text
NO DATA BUILD WITHOUT READING ITS QUALITY REPORT —
FIX THE SYNTHESIS HARNESS, NEVER THE GATES
```

`build_train_data` runs strict-by-default (fuzzy + semantic dedup, tier floor,
n-gram decontamination vs eval suites, exposure caps) and every build emits
`quality_report.json`, `rejected.jsonl` (nothing dropped silently), and
`synthesis_feedback.json` (per-family/synthesizer yields, recommendations,
autoresearch-shaped experiment candidates). After any build: read the
feedback, act on the named producer/synthesizer, file the experiment
candidates — **REQUIRED SKILL:** `synthesis-feedback`. `--profile permissive`
is a diagnostic escape hatch, never a fix; gate/threshold changes go through
`honest-ship-eval`. Cross-snapshot overlap is audited with
`scripts/audit_data_corpora.py` (durable results in
`docs/design/data-corpus-audit.*`); exclude covered pairs with
`--dedup-against`. Runs bind to their exact dataset (`data_manifest_sha` ↔
lineage `DataSnapshot`); derived curation uses `--derive-from`,
`--difficulty-from` (record NLL evidence), and
`scripts/mine_rejected_preferences.py`.

| Excuse | Reality |
| --- | --- |
| "outputs/ is enough" | Reviewers read `docs/design/`. |
| "JSON written; markdown later" | Headline tables are the scoreboard. |
| "Failed/partial — skip docs" | Document failure + recipe. |
| "It's in the PR body" | PR text is ephemeral. |
| "Bucket URI is enough; skip the model card" | Card + README summary are how humans find the checkpoint. |
| "The code SHA is enough" | Component versions are what make results comparable and retestable. |

**REQUIRED SKILL:** `documenting-experiment-results`.

## Normalized component versioning

The eval/smoke/checkpoint stack is self-improving, so every result must say
which revision of the constraints produced it. Contract:
`docs/design/version-stamp-contract.md`.

- **Registry:** `src/slm_training/resources/versions.json` maps component ids
  (`harness.model_build.eval`, `evals.meaningful_program`, `gates.ship`,
  `matrix.quality`, …) to their current version, watched `paths`, and an
  append-only `history` (newest first).
- **Stamp:** canonical writers embed a `version_stamp` envelope
  (`stamp_schema: version_stamp/v1` — code commit, dirty flag, component
  versions, timestamp) in every eval/scoreboard/gates/matrix/bench/train
  payload via `slm_training.versioning.build_version_stamp`.
- **Bump rule:** changing a watched file requires touching that component's
  registry entry in the same change — a version bump (new ids use monotonic
  `v1, v2, …`) or a same-version history entry whose note starts with
  `no-bump:` for behavior-neutral edits. Enforced by
  `python -m scripts.verify_version_stamps --check` (CI, pre-commit, agent
  hooks).
- **Re-test discovery:** after a bump, `python -m scripts.verify_version_stamps
  --stale [--component <id>] [--include-outputs]` lists results produced under
  older constraints — the candidates worth re-running. Experiments that ran
  against since-fixed constraints stay valuable; keep them discoverable, never
  silently comparable.

## Engineering norms

- Prefer harness/script changes over one-off notebooks.
- Preserve train/test isolation and structural leakage checks.
- Never reintroduce silent `gold.placeholders` channels under
  `honest_slot_contract=True`.
- Authoritative deterministic decode proofs always outrank learned, semantic,
  confidence, or preference scores. Commit an exact legal singleton before any
  neural ranking and never downgrade certainty into a soft preference.
- New production generation/decoder paths must project model-facing symbol identity
  to stable opaque request-local ordinals. External names, template-marker spellings,
  and alias-derived text, hashes, or embeddings are codec/realization data and must
  not become scoring or legal authority. Typed role/type metadata may be supplied
  separately as declared authority; restore caller names only after verified decode.
  Historical default-off name-aware experiment modes may remain for checkpoint and
  evidence compatibility, but must not become a production default or new authority.
- Say fixture-demo vs ship. Do not weaken ship gates to green CI.
- Match existing style; no unrelated drive-by refactors.
- Before adding or relocating tracked paths, use `organize-repository`, follow
  `docs/repository-organization.md`, and use `git mv` rather than `mv` for moves.
- Frozen DSL-agnostic harness machinery (version stamping, lineage records,
  checkpoint references, the gate/promotion engines, scaling math, eval
  bookkeeping) lives in `src/slm_training/harness_core/` and never imports the
  DSL/model/eval/harness layers; DSL specifics enter via callbacks. Old import
  paths are stable shims. Changes there bump the `harness.core` component
  (see `docs/design/harness-core.md`).

```
docs/MODEL_CARD.md # checkpoint roster + eval (keep README summary in sync)
docs/design/       # matrices + measured results (source of truth)
scripts/           # train / eval / matrix / bench CLIs
src/slm_training/  # implementation (harness_core/ = frozen DSL-agnostic core)
.agents/skills/    # canonical skills for all tools
```

<!-- OPENWIKI:START -->

## OpenWiki

This repository uses OpenWiki for recurring code documentation. Start with `docs/openwiki/quickstart.md`, then follow its links to architecture, workflows, domain concepts, operations, integrations, testing guidance, repository organization, and source maps.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
