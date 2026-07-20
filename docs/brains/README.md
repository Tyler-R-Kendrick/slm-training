# Brains — curated research knowledge (OKF + Obsidian)

This tree is the **human-curated knowledge layer** for the OpenUI SLM research
program. It complements — never duplicates — the machine-generated OpenWiki
(`docs/openwiki/`) and the measured evidence under `docs/design/`.

The `autoresearch` skill reads and updates this tree as the memory of the
research loop. Start there: [`.agents/skills/autoresearch/SKILL.md`](../../.agents/skills/autoresearch/SKILL.md).

## Three knowledge surfaces (do not confuse them)

| Surface | Path | Authored by | Role |
| --- | --- | --- | --- |
| **OpenWiki** | `docs/openwiki/` | `openwiki` CLI (generated) | Navigation over the *current* codebase; refreshed, not hand-edited |
| **Design evidence** | `docs/design/` | experiments (`documenting-experiment-results`) | Source of truth for *measured* results; `iter-*.md` ↔ Linear `SLM-N` |
| **Brains** | `docs/brains/` | humans + `autoresearch` | Durable *ideas, syntheses, open questions, lineage* across experiments |

Rule of thumb: if a fact is **measured**, it lives in `docs/design/`; if it is
**derivable from code**, it lives in OpenWiki; if it is an **idea, synthesis,
hypothesis seed, or connection between the two**, it lives in a brain.

## OKF — the Open Knowledge Framework

Brains follow **OKF**, the repo's Obsidian-compatible note convention:

1. **Atomic notes.** One concept, hypothesis, or source per file. Small and
   composable, not chapter-length.
2. **Typed frontmatter.** Every note opens with YAML: `type`, `status`, `tags`,
   `created`, and cross-links (`linear`, `design`, `sources`). See
   [`templates/`](templates/).
3. **Wikilinks.** Connect notes with Obsidian `[[wikilinks]]`; connect out to
   repo files with normal relative Markdown links whose target is a repo path
   (such as `docs/design/` or `docs/openwiki/`).
   Wikilinks are the graph; Markdown links are the anchor to ground truth.
4. **Maps of Content (MOCs).** Index notes (`*/MOC.md`) are the entry points —
   the graph is navigated top-down from a MOC, not by folder listing.
5. **Evidence-linked.** A claim note must link the `docs/design/` record or the
   external source that backs it. Unbacked claims are marked `status: seed`.
6. **No leakage.** Brains never embed held-out eval content, secrets, or
   machine-absolute paths (repo policy rejects the last one in `docs/design/`).

## Layout

```text
docs/brains/
  README.md            # this file — OKF conventions
  repo/                # the shared repo brain (versioned, reviewed)
    MOC.md             # top-level map of content
  personal/            # personal brains maintained in the repo
    README.md          # how to attach an Obsidian vault as a personal brain
  templates/           # OKF note templates (Obsidian "Templates" folder)
    concept-note.md
    source-note.md
    experiment-idea.md
```

Open `docs/brains/` as an Obsidian vault (or add it to an existing vault as a
folder) to edit with backlinks, graph view, and templates. The
`templates/` folder is a ready-made Obsidian *Templates* source.
