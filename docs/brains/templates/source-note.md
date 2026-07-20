---
type: source
status: captured     # captured | summarized | mapped
tags: [source]
created: {{date}}
citation:            # authors, title, venue/year
url:                 # arXiv / page URL
fidelity:            # Faithful | Adapted | Surrogate | Adjacent (see research-lineage.md)
lineage:             # research-lineage.md section or source-manifest id, if mapped
---

# {{title}}

## One-line

What the source claims, in a sentence.

## What we could take

The specific idea/mechanism relevant to OpenUI TwoTower / grammar-diffusion.

## What we would NOT take

Boundary — the parts that do not transfer (and why).

## Connections

`[[concept-note]]`s this supports or challenges; related `[[source-note]]`s.

> When a source graduates from a brain note into implemented lineage, record it
> in [`docs/design/research-lineage.md`](../../design/research-lineage.md) or the
> matching `src/slm_training/resources/autoresearch/*.json` manifest and link
> back here. The brain note is the scratchpad; lineage is the ledger.
