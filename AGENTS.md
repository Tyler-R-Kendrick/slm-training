# Agent instructions (all coding agents)

Applies to **every** coding agent (Cursor, Claude Code, Codex, Gemini, Copilot /
GHCP, Grok, others). Prefer this file over tool-specific defaults on process
conflicts.

Every harness enforces the **same** laws. `python -m scripts.verify_agent_surfaces`
owns the obligation × surface matrix and fails CI when one surface drops a law;
current coverage and known differences are in
[`docs/design/agent-harness-parity-audit.md`](docs/design/agent-harness-parity-audit.md).

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

## Non-negotiable architecture invariants (goal law)

This repo is **not training a natural-language LLM.** It trains a
**grammar-constrained symbolic diffusion model** that outputs templated
grammars (scaffolded structure + structural reasoning). Templated /
natural-language content is deferred to a real external LLM (symbol-only output
contract, `dsl/language_contract.py`, `OUTPUT_CONTRACT_VERSION=2`). Everything
below is a **goal invariant**: it constrains every model, harness, lever,
experiment, doc, and agent action here — past, present, and future. Canonical
expansion with file pointers, current status, and open goals:
[`docs/design/decode-invariants.md`](docs/design/decode-invariants.md).

Ids below are the canonical `I*` ids from
[`docs/design/decode-invariants.md`](docs/design/decode-invariants.md). Cite
them, never a file-local number — every agent surface uses the same ids so
"I13" means one thing everywhere.

### I. Constrained decoding is the product (never removable)

- **I6 — Never output invalid grammar.** Every production decode path is
  grammar-constrained end to end. Unconstrained arms
  (`--unconstrained-control`, HTTP `grammar_constrained=false`, eval `raw`
  arms) are **diagnostic controls only** — never production defaults, never
  serving paths, and their output is never shipped, certified, or gated on.
- **I6 — Fail closed.** Production configs must not set
  `allow_unconstrained_fallback=True`, must run finalize validation, and must
  raise (not return uncertified text) when certification fails — in every
  backend, ONNX included. An empty legal domain is a constrained dead end,
  never a full-vocabulary fallback. This extends past decode: no load-bearing
  contract may silently widen when a dependency is unavailable.
- **I6 — No lever, experiment, or skill may remove or weaken deterministic /
  constrained decoding.** Levers may change *how legal symbols are chosen*
  (ranking, speculation technique, batching) — never *whether output is
  legal*. Weakening levers are registered in
  `levers.CONSTRAINT_WEAKENING_LEVERS` and CI-blocked from production configs.

### II. Inference is the last resort (deterministic completion law)

- **I1 — Deterministic completion paths bypass inference wherever a
  deterministic answer exists.** Authoritative deterministic decode proofs
  always outrank learned, semantic, confidence, or preference scores.
- **I2 — Forced bypass on singletons:** when the scope-aware symbol table (DFA
  domain / `CompletionDomainV1` / choice-codec state) shows **exactly one**
  valid next symbol, that symbol is committed **without any neural forward or
  ranking**, in every decode path and every backend. Never downgrade certainty
  into a soft preference. New decode paths ship with a bypass test (the
  `forwards_count == 0` pattern) or they do not merge.
- **I3 — Speculative completion from forward-calculated symbol tables:** symbol
  tables are computed *before* the model; at non-singleton branch points, rank
  legal symbols with a deterministic scorer
  (`dsl/grammar/fastpath/speculative_rank.py`, committed train-only n-gram
  table at `resources/decode/speculative_ngram_v1.json`) and speculatively commit
  multi-token spans that stay inside the certified domain (lookahead-then-
  verify, arXiv:2602.00612; intersection-witness completions,
  arXiv:2508.10111). **I5 —** the *technique* is a lever, swappable by
  preregistered experiment, but speculation always verifies against the grammar
  oracle before commit.
- **I4 — Symbol tables schedule compute:** use them to plan subsequent prefills —
  compact ambiguous rows into minimal forwards, place prefill boundaries at
  grammar checkpoints proven by `common_forced_run` (what the grammar forces
  after *every* legal candidate is determined before the model picks), route
  by detected device (`runtime/decode_schedule.py`). Record scheduled-prefill
  and forwards-avoided counters in `DecodeStats`; utilization regressions are
  measured, never vibes.

### III. What the model is (and is not)

- **I9 — Output = scaffolded grammar.** Targets contain only grammar/AST literals
  and placeholder symbols. Natural-language vocab is **optional fluff** —
  optimizable later, never load-bearing, never a ship blocker.
- **I10 — Use-case ladder (in order, no skipping):** AST-2-AST → grammar-2-AST →
  grammar+ops-2-AST → simplified-NL-2-AST → complex-NL-2-AST. Each rung is
  certified before the next opens (`CERT_CAP*` gates stay). Current position
  and blockers: `docs/MODEL_CARD.md` + `docs/design/decode-invariants.md`.
- **I10b — Calculator/solver enhanced with inference** — not a chat model.
  Inference fills ambiguity; it never authors structure a deterministic solver
  can derive.

### IV. Encoder/decoder vocabulary law

- **I13 — The encoder vocabulary MUST reserve a compute-ops vocabulary** —
  AST/graph/set/topology operations — **known and shared by the decoder
  vocabulary**. That vocabulary is `dsl/ops_vocab.py`: derived from the live
  operator registries (an op cannot be in it without an implementation, or
  implemented without being in it), reserved in the versioned `ops` token-id
  namespace, and exposed through the single `shared_token_ids()` mapping both
  towers call. Grammar symbols layer on top via `assert_layering`; NL sits
  above and is strictly optional. Adding, removing, or reclassifying an
  operator changes the fingerprint and fails
  `verify_decode_invariants` until `resources/ops_vocab_registry.json` is
  rebuilt and `ops.vocab` is bumped. e803 rejected *decoder-target* op tokens
  and says nothing about encoder-side sharing; that campaign is the open rung.
  The output tokenizer layout is likewise a frozen, checkpoint-bound contract
  (`resources/tokenizer_layout_registry.json`) — never re-derived from whichever
  grammar backend happens to be live.
- **I11 — Multi-turn = CRDT event store.** Append-only, content-addressed events
  over the conversation AST (`ConversationTraceV1`). Turn inputs are ops on
  that AST; ops include **copy/undo/redo**. Merge must converge (CRDT
  semantics) — the conflict-rejecting merge is a documented interim state,
  not the goal. The AST artifact is a **materialization of the entire
  conversation history** (full replay, no hidden cursors).
- **I12 — Patch/diff outputs across turns.** Turns emit operation patches/diffs,
  not full rewrites, wherever the edit space can reach the target
  (reachability-certified). Full-AST output is the bootstrap mode, not the
  end state; reachability blockers are open goals, not closed questions.

### V. Goal-drift guard

- **I14 — Goals are non-negotiable; approaches are disposable.** A rejected
  experiment closes an *approach*, never a *goal*. Every rejected approach to
  an invariant above must file its successor approach (or an explicit, dated,
  documented waiver) in the same measured-results doc. Labels like
  "rejected" / "unavailable" / `nl_available=False` /
  `reachable_fraction=0.0` describe **current approach state** and may never
  be cited as reason the invariant does not apply.
- **I7 — Every agent surface carries the law.** Each configured harness reads a
  different instruction file, so a law stated only here reaches only some
  agents. `python -m scripts.verify_agent_surfaces` owns the obligation ×
  surface matrix and certifies every law on every surface. See
  [`docs/design/agent-harness-parity-audit.md`](docs/design/agent-harness-parity-audit.md).
- **I15 — Everything is documented.** These invariants live canonically in
  `docs/design/decode-invariants.md`, are linked from README, MODEL_CARD, and
  the decode/vocab/conversation design docs, and are regenerated into
  OpenWiki. Changing one requires editing that doc, bumping the
  `decode.invariants` component in `resources/versions.json`, and passing
  `python -m scripts.verify_decode_invariants` in CI — a silent weakening is
  a regression and blocks merge.

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

Refresh / reinstall: **[`.agents/skills/README.md`](.agents/skills/README.md) is
the single owner** of those commands. Do not copy them here — the two lists had
already diverged, and the marketplace installers need cleanup steps
(`--copy` leaves real directories where `repo_policy` requires symlinks) that
only the README carries.

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

Already installed under `.agents/skills/` and symlinked for Cursor/Claude.
Refresh commands and their cleanup steps live in
[`.agents/skills/README.md`](.agents/skills/README.md) — the single owner.

Cursor also loads MCP from [`.cursor/mcp.json`](.cursor/mcp.json) (Playwright +
Hugging Face Hub MCP + **Serena**). Optional UI install:
[Cursor marketplace — Hugging Face](https://cursor.com/marketplace/huggingface).

### MCP servers are not uniform across harnesses

Skills are identical everywhere; MCP servers are not. Check before assuming a
server is available, and fall back to the CLI (`hf`, `npx playwright`) when it
is not.

| Server | Claude Code | Cursor | VS Code / Copilot Chat | Codex |
| --- | :-: | :-: | :-: | :-: |
| Serena | ✅ [`.mcp.json`](.mcp.json) | ✅ | ✅ [`.vscode/mcp.json`](.vscode/mcp.json) | ✅ [`.codex/config.toml`](.codex/config.toml) |
| Linear (`autoresearch` issue filing) | ✅ | — | — | — |
| Playwright (`playwright-cli`) | — | ✅ | — | — |
| Hugging Face Hub | — | ✅ | — | — |

Linear is Claude-only because `autoresearch` issue/milestone filing runs there;
Playwright and the HF Hub server are Cursor-only because both have first-class
CLIs (`npx playwright`, `hf`) that every other harness uses instead.

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
| Codex | [`.codex/config.toml`](.codex/config.toml) (committed) + hooks in [`.codex/hooks.json`](.codex/hooks.json); older builds need [`.codex/serena.config.toml.example`](.codex/serena.config.toml.example) → `~/.codex/config.toml` (or `serena setup codex`) |
| Copilot CLI | `/mcp add` → `serena start-mcp-server --context=copilot-cli --project-from-cwd`; hooks in [`.github/hooks/`](.github/hooks/) |

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

## Preregistered campaign law

AP-007+ experiment runners and every promotion candidate use the canonical
`ExperimentCampaignV1` contract in
`src/slm_training/autoresearch/experiment_campaign.py`. Lock the manifest in
the campaign event chain before execution; bind plans, outcomes, and promotion
evidence to that digest. Deviations are append-only and exploratory. Never
replace the locked confirmatory endpoint, arms, seeds, stopping rule, family,
or gates after outcomes are visible. Meaning-v2 becomes the default primary
only after a hash-verified AP-001 `certified` artifact; otherwise use the
binder/reference F1 fallback. See
[`docs/design/experiment-campaign-governance.md`](docs/design/experiment-campaign-governance.md).

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
