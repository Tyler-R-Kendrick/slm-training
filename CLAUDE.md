# Claude Code instructions

Follow **[AGENTS.md](AGENTS.md)** — the canonical instructions for every coding
agent in this repo.

Obey **AGENTS.md § Non-negotiable architecture invariants** — constrained
decoding is the product, deterministic/singleton bypass outranks any learned
score, speculation ranks only over forward-calculated symbol tables and always
verifies before commit, symbol tables schedule prefills, the ops vocabulary is
shared encoder↔decoder, and multi-turn is a CRDT event store. Capability is
never bought with parameters: size is a charged budget, growth needs
`EG_params` ≥ 1, promote the smallest sufficient model, and scaling is a
diagnostic control arm — never a default lever. Never weaken
them; a rejected experiment closes an approach, never a goal. Canonical
expansion: [`docs/design/decode-invariants.md`](docs/design/decode-invariants.md).

Load skills from `.agents/skills/` (mirrored / symlinked under `.claude/skills/`).
Use `organize-repository` before changing tracked file placement and use
`git mv` for every tracked relocation.
After any train / eval / benchmark / matrix run, use
`documenting-experiment-results`. After any training-data build or synthesis,
use `synthesis-feedback` — read the build's `quality_report.json` /
`rejected.jsonl` / `synthesis_feedback.json` and improve the synthesis
harness from that evidence (never weaken the gates). Token stack: `ponytail`, `caveman`,
`headroom`, `rtk` (see `AGENTS.md` / `RTK.md`). Hugging Face pack:
`hf-cli` + marketplace skills from
[huggingface/skills](https://github.com/huggingface/skills)
(`hf skills add --claude --force` / `hf skills update` to refresh).
When a checkpoint is created or promoted, update `docs/MODEL_CARD.md` and the README model-card summary.
Version stamps: results carry `version_stamp`; changing a metric/gate/harness/matrix
file requires a component bump (or `no-bump:` note) in
`src/slm_training/resources/versions.json` — see AGENTS.md “Normalized component
versioning” and `docs/design/version-stamp-contract.md`.
When you change a dashboard page (`src/apps/dashboard/src/pages/*.tsx`), keep its interpreted-mode
`src/slm_training/web/static/openui/*.openui` program at parity and run `scripts/validate_page_dsl.py` —
use the `dashboard-openui-parity` skill.

Serena MCP (semantic code tools) is configured for this repo — prefer Serena
symbol tools over raw grep when navigating `src/` / `scripts/`. See `AGENTS.md`
and `.serena/project.yml`.

<!-- OPENWIKI:START -->

## OpenWiki

This repository uses OpenWiki for recurring code documentation. Start with `docs/openwiki/quickstart.md`, then follow its links to architecture, workflows, domain concepts, operations, integrations, testing guidance, and source maps.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
