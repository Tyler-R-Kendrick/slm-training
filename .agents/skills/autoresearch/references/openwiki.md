# OpenWiki — generated codebase knowledge

OpenWiki (`docs/openwiki/`) is the machine-generated navigation layer over the
*current* code. The loop reads it to ground hypotheses in what the repo actually
does, and refreshes it when the surfaces involved have drifted. It is generated —
do not hand-edit generated pages; change source/docs and regenerate.

## Read

Entry point: [`docs/openwiki/quickstart.md`](../../../../docs/openwiki/quickstart.md),
then its links (architecture, workflows, operations, testing, source map). Use it
to answer "where does X live / how does the pipeline flow" before proposing a
change, so a hypothesis targets a real owner, config knob, and artifact root.

## The OpenWiki CLI

Wrapped by [`scripts/update_openwiki.py`](../../../../scripts/update_openwiki.py),
which keeps generated pages under `docs/openwiki/` and forwards its args verbatim
to the external `openwiki code` binary (so flags like `--init` / `--update` /
`--print` are the pinned `openwiki@0.1.2` CLI's, not this repo's):

```bash
npm install -g openwiki@0.1.2                      # once
python -m scripts.update_openwiki --init --print   # first generation (needs a provider key)
python -m scripts.update_openwiki --update --print  # refresh after code/doc changes
```

Provider secrets: prefers `OPENAI_API_KEY`, then `OPENROUTER_API_KEY` (optional
`LANGSMITH_API_KEY` enables tracing). A scheduled refresh runs via
[`.github/workflows/openwiki-update.yml`](../../../../.github/workflows/openwiki-update.yml).
Generation instructions live in
[`docs/openwiki/INSTRUCTIONS.md`](../../../../docs/openwiki/INSTRUCTIONS.md) — do
not rewrite that file during routine `--update`.

## When the loop touches OpenWiki

- **Refresh** when a surface a hypothesis targets has changed since the last
  generation, or the scheduled workflow has not run since a relevant merge.
- **Do not** commit hand edits to generated pages; update the source code/docs
  and let `--update` regenerate.
- **Brains vs OpenWiki:** OpenWiki tells you *what the code is*; the brain records
  *what we think about it and want to try*. Link OpenWiki pages from brain notes;
  never paste generated navigation into a brain.

Approvals: `--init` / `--update` call an external LLM provider — only run them
with a configured provider secret and within the user's cost expectations.
