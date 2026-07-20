# Brains — read & update (OKF / Obsidian)

The brains under [`docs/brains/`](../../../../docs/brains/) are the memory of the
research loop. Read them before hypothesizing; update them after results. Full
conventions: [`docs/brains/README.md`](../../../../docs/brains/README.md).

## OKF in one screen

- **Atomic, typed notes.** One idea per file, opening with YAML frontmatter
  (`type`, `status`, `tags`, `created`, and cross-links `linear` / `design` /
  `sources`). Use the templates in
  [`docs/brains/templates/`](../../../../docs/brains/templates/).
- **Wikilinks build the graph; Markdown links anchor to ground truth.** Connect
  notes with Obsidian `[[wikilinks]]`; link out to repo files with relative
  Markdown links.
- **MOCs are the entry points.** Navigate from
  [`docs/brains/repo/MOC.md`](../../../../docs/brains/repo/MOC.md), not by folder
  listing.
- **Every claim is evidence-linked** or marked `status: seed`.

## Two brains

- **Repo brain** (`docs/brains/repo/`): shared, reviewed consensus knowledge.
- **Personal brains** (`docs/brains/personal/<owner>/`): individual WIP thinking;
  same OKF rules. Promotion path and ownership:
  [`docs/brains/personal/README.md`](../../../../docs/brains/personal/README.md).

## Read (before hypothesizing)

1. Open the repo brain MOC; follow wikilinks to the notes for the objective.
2. Read the objective's **open questions** and **dead-ends** sections — these
   bound discovery and gate the novelty audit.
3. Read any personal-brain notes the owner points at for the objective.

## Update (after results / discovery)

- **New idea** → `templates/concept-note.md` (or `experiment-idea.md` if it is
  run-ready); wikilink it from the MOC.
- **New source** → `templates/source-note.md`; record what to take / not take and
  the fidelity label. When a source graduates to implemented lineage, record it
  in `research-lineage.md` / the source manifest and link back.
- **Answered question** → move it out of "open questions" and link the
  `docs/design/iter-*.md` row that answered it.
- **Negative result** → add to "dead-ends" with the evidence link so the
  hypothesis novelty audit excludes it.

## Guardrails

- Brains hold **knowledge, not measured results** — those live in `docs/design/`.
  A brain that restates a scoreboard is drifting; link instead.
- No secrets, credentials, machine-absolute paths, or held-out eval content.
- Keep the graph connected across moves (update backlinks when promoting a note).
- Optional Obsidian: open `docs/brains/` as a vault for backlinks / graph view;
  the loop works from plain Markdown regardless.
