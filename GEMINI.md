# Gemini CLI instructions

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

Activate skills from `.agents/skills/`. After any train / eval / benchmark /
matrix run, use `documenting-experiment-results`. Token stack: `ponytail`,
`caveman`, `headroom`, `rtk` (see `AGENTS.md` / `RTK.md`). Hugging Face pack:
`hf-cli` + marketplace skills from
[huggingface/skills](https://github.com/huggingface/skills)
(`hf skills add --force` / `hf skills update` to refresh).
Use `organize-repository` before changing tracked file placement and use
`git mv` for every tracked relocation.
When a checkpoint is created or promoted, update `docs/MODEL_CARD.md` and the README model-card summary.
