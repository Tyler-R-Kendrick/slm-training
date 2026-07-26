# Gemini CLI instructions

Follow **[AGENTS.md](AGENTS.md)** — the canonical instructions for every coding
agent in this repo. Prefer it over tool-specific defaults on process conflicts.

Obey **AGENTS.md § Non-negotiable architecture invariants** — constrained
decoding is the product, deterministic/singleton bypass outranks any learned
score, speculation ranks only over forward-calculated symbol tables and always
verifies before commit, symbol tables schedule prefills, the ops vocabulary is
shared encoder↔decoder, and multi-turn is a CRDT event store. Never weaken
them; a rejected experiment closes an approach, never a goal. Canonical
expansion: [`docs/design/decode-invariants.md`](docs/design/decode-invariants.md).

Activate skills from `.agents/skills/`.

## Repository laws (identical for every agent)

Parity across harnesses is enforced by
`python -m scripts.verify_agent_surfaces`; see
[`docs/design/agent-harness-parity-audit.md`](docs/design/agent-harness-parity-audit.md).

- **Hard run cap.** Every train / eval / bench / profile / matrix run and its
  supporting shell commands obey `MAX_RUN_MINUTES` in
  `src/slm_training/levers.py`. A timed out or killed run is never evidence.
- **Iron law: docs follow every experiment.** After any train / eval /
  benchmark / profile / telemetry / matrix / reproduction run, use
  `documenting-experiment-results`. Numbers only in `outputs/`, chat, or a PR
  comment are incomplete work.
- **Honest ship gates.** Use `honest-ship-eval` when evaluating, writing or
  interpreting gates, or claiming readiness. Say fixture-demo vs ship; never
  weaken a gate to green CI.
- **Data-quality law.** After any training-data build or synthesis, use
  `synthesis-feedback` — read `quality_report.json`, `rejected.jsonl`, and
  `synthesis_feedback.json`, and fix the synthesis harness, never the gates.
- **Model card.** When a checkpoint is created, synced, bootstrapped, or
  promoted, update `docs/MODEL_CARD.md` **and** the README model-card summary.
- **Version stamps.** Results carry `version_stamp`; changing a watched
  metric / gate / harness / matrix file requires a component bump (or a
  `no-bump:` history note) in `src/slm_training/resources/versions.json` —
  `python -m scripts.verify_version_stamps --check` enforces it.
- **Dashboard parity.** When you change a dashboard page
  (`src/apps/dashboard/src/pages/*.tsx`), keep its interpreted-mode
  `src/slm_training/web/static/openui/*.openui` program at parity and run
  `scripts/validate_page_dsl.py` — use the `dashboard-openui-parity` skill.
- **Preregistered campaigns.** Experiment runners and promotion candidates use
  the `ExperimentCampaignV1` contract; never replace a locked confirmatory
  endpoint, arms, seeds, stopping rule, or gates after outcomes are visible.
- **Repository organization.** Use `organize-repository` before creating,
  moving, renaming, or deleting tracked paths, and use `git mv` for every
  tracked relocation.

Token stack: `ponytail`, `caveman`, `headroom`, `rtk` (see `AGENTS.md` /
`RTK.md`). Hugging Face pack: `hf-cli` + marketplace skills from
[huggingface/skills](https://github.com/huggingface/skills)
(`hf skills add --force` / `hf skills update` to refresh).

Gemini CLI has no committed hook configuration in this repo, so the raw-`mv`
guard and post-edit checks that Claude Code and Codex get do not run for you.
Run them yourself before finishing: `python -m scripts.repo_policy` and
`.githooks/check-changed`.

<!-- OPENWIKI:START -->

## OpenWiki

This repository uses OpenWiki for recurring code documentation. Start with `docs/openwiki/quickstart.md`, then follow its links to architecture, workflows, domain concepts, operations, integrations, testing guidance, and source maps.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
