---
type: moc
status: living
tags: [moc, repo-brain]
created: 2026-07-20
---

# Repo brain — Map of Content

Top-level entry point for the shared repo brain. Everything here links **out**
to ground truth (design evidence, OpenWiki, Linear) and **across** to other
notes via `[[wikilinks]]`. Keep this map short; push detail into atomic notes.

## Ground-truth anchors

- Measured evidence: [`docs/design/`](../../design/) — `iter-*.md` ↔ Linear `SLM-N`
- Research lineage (cited papers → code): [`docs/design/research-lineage.md`](../../design/research-lineage.md)
- Autoresearch / autotraining spec: [`docs/design/autoresearch-autotraining.md`](../../design/autoresearch-autotraining.md)
- Source manifests: [`src/slm_training/resources/autoresearch/`](../../../src/slm_training/resources/autoresearch/)
- Codebase navigation: [OpenWiki quickstart](../../openwiki/quickstart.md)
- Model card: [`docs/MODEL_CARD.md`](../../MODEL_CARD.md)

## Active theses (Linear initiatives / projects → brain notes)

Each active Linear project should have (or grow) an atomic note here capturing
the *thesis*, the open questions, and the falsification boundary. Seed notes:

- _(seed)_ Valid-edit flow attribution — link project + `docs/design/` rows
- _(seed)_ Semantic planning & valid-state learning
- _(seed)_ Calculated arity & adaptive precision
- [[gate-reachability-and-power]] — separate evidence volume and measurement
  integrity from model quality; freeze power inputs before confirmation.
- [[preregistered-experiment-campaigns]] — bind execution and promotion to one
  decision-complete pre-outcome manifest; keep deviations exploratory.
- [[decode-invariants-goal-law]] — constrained decoding, deterministic bypass,
  symbol tables as ranker *and* scheduler, shared ops vocab, CRDT multi-turn;
  a rejected approach never closes a goal.

> Add a note with `templates/concept-note.md` and wikilink it here when a thesis
> becomes active. Do not restate the Linear project — link it and record what the
> repo has *learned* about it.

## Open questions (drives hypothesis generation)

Curate the live "what don't we know yet" list here. The `autoresearch` loop
reads this section to seed the autotrain hypothesis loop and prunes it as
`docs/design/` rows resolve questions.

- _(seed)_ Does discrete flow matching add anything beyond the legal-edit state
  space + complete bridge supervision? (see valid-edit flow project)
- [[recursive-depth-supervision-objective]] — arithmetic/compatibility contract
  is supported; real-suite mode selection remains open.
- [[recursive-recurrence-health]] — **updated 2026-07-25 (SLM-421)**: the n=2
  SLM-282 negative was underpowered noise, not a robust contraction
  violation. A powered n=20 Wilson-interval rerun (fresh disjoint seeds,
  preregistered `min_pass_rate=0.5` locked before observing them) found
  `recursive_core_positive` (18/20, Wilson 95% CI [0.699, 0.972]). This
  satisfies only one of PR #853-#856's two LAR3 reopening conditions; LAR3
  stays closed pending a powered rerun of SLM-317's value gate (blocked —
  that harness is not yet merged to `main`). Next open question: land
  SLM-317's harness on `main` (via its own PR #852) so its value gate can
  get the same powered-rerun treatment.

## Dead ends (do not re-propose)

Record negative results and abandoned levers with a link to the evidence, so the
hypothesis loop's novelty audit can exclude them.

- _(seed)_ E249 / E252 local-preference chain — negative; see
  [`research-lineage.md`](../../design/research-lineage.md) "Exact-state local decision preference".
