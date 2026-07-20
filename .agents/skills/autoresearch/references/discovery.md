# Discovery — prior work & related research

Find related research and in-repo lineage for the objective *before*
hypothesizing, so every hypothesis is grounded and the novelty audit has
material. Capture repository evidence first, then reach outside.

## In-repo first (cheapest, highest signal)

1. **Lineage ledger.** Read
   [`docs/design/research-lineage.md`](../../../../docs/design/research-lineage.md)
   — the source of truth for cited papers, fidelity labels
   (Faithful / Adapted / Surrogate / Adjacent), and where each maps in code.
2. **Source manifests.**
   [`src/slm_training/resources/autoresearch/*.json`](../../../../src/slm_training/resources/autoresearch/)
   hold normalized per-campaign source inventories (metadata, summaries,
   applicability, limitations, fidelity). Reuse them; do not re-summarize.
3. **Prior runs.** Grep `docs/design/iter-*.md` for matched controls and prior
   negative results on the same lever (these become dead-ends / controls).

## Campaign research loop (delegated — this skill does not run it)

When the objective needs the autoresearch harness's own literature pass, that is
the campaign `research` step, owned by **`openui-autoresearch`** (commands and
contracts there) and run via `autotrain`. This skill *consumes* the persisted
result — the canonical bundle `outputs/autoresearch/<campaign>/` and any
`source` notes it produced — it does not re-issue `init` / `research` itself.

## External research (when in-repo is exhausted)

For a broad, fact-checked external sweep use the `deep-research` skill or web
search + `WebFetch`, and for library/API specifics use Context7. Then:

- Write a `source` note per genuinely new source
  ([brains.md](brains.md), `templates/source-note.md`) with take / don't-take and
  a fidelity label.
- If a source is implementation-worthy, stage its promotion into
  `research-lineage.md` / a source manifest — the brain note is the scratchpad,
  lineage is the ledger.

## Output of this stage

- Updated `source` notes + refreshed **open questions** and **dead-ends** in the
  brain.
- A shortlist of grounded, cited levers ready for the hypothesis loop — each with
  a fidelity label and an explicit "what would falsify it" line.

## Guardrails

- Do not treat several adjacent papers as independent evidence for one lever
  (the lineage doc is explicit about this).
- Inherited results do not transfer: reported speedups / quality gains from a
  paper are lineage labels, never repo claims.
