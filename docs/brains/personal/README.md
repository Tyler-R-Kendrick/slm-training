# Personal brains

Personal brains are individually-owned Obsidian vaults maintained **inside the
repo** so the `autoresearch` loop can read and update them alongside the shared
repo brain. They hold work-in-progress thinking that is not yet consensus repo
knowledge.

## Attach a personal brain

`<owner>` is the owner's short handle — their GitHub username or the short name
they use in the repo (the same identity that authors commits). The
`autoresearch` loop resolves it from the current user's handle; if it cannot, it
asks rather than guessing. A committed illustration lives in
[`example/home.md`](example/home.md).

Create a folder named for the owner and open it (or the whole `docs/brains/`
tree) as an Obsidian vault:

```text
docs/brains/personal/<owner>/
  home.md            # the vault's home note / dashboard (OKF: type: moc)
  concepts/          # atomic concept notes
  sources/           # source notes (papers, threads, videos)
  ideas/             # experiment-idea notes staged for the hypothesis loop
```

Use [`../templates/`](../templates/) as the Obsidian *Templates* source so every
note gets typed OKF frontmatter.

## Contract

- **Same OKF rules** as the repo brain (atomic, typed frontmatter, wikilinks,
  evidence-linked, no leakage/secrets/absolute paths).
- **Promotion path.** When a personal note becomes shared knowledge, the
  `autoresearch` loop promotes it into `docs/brains/repo/` and wikilinks it from
  [`repo/MOC.md`](../repo/MOC.md). A promotion is a tracked relocation: follow the
  `organize-repository` skill and use `git mv` for the move, then verify the
  backlinks/wikilinks still resolve so the graph stays connected across it.
- **Ownership.** Only the owner (or `autoresearch` acting on their behalf) edits
  a personal vault; others link to it, they do not rewrite it.

> A personal vault may instead live outside the repo and be symlinked in, but the
> committed default is an in-repo folder so the loop and reviewers can see it.
