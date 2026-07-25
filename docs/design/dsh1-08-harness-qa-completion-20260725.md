# DSH1-08: capped identity, verified COMPLETE_SUFFIX accepted sets, anti-identity negatives (SLM-360)

**Issue:** SLM-360 (project DSH1 — CAP0 Grammar-Complete Symbolic Synthesis,
milestone M3 — Symbolic QA views and legal partials)
**Status:** wiring / unit-tested contract. No decode-path integration, corpus
build, training run, eval, benchmark, checkpoint, or ship claim.

## Decision

Create CAP0 QA views that test copy, normalization, and completion without
allowing identity examples or arbitrary single-gold derivations to dominate:
cap identity rows and mark them diagnostic, generate verified Harness-DSL
`COMPLETE_SUFFIX` questions whose accepted-completion set is fully persisted
(never collapsed to one arbitrary gold), require every atomic form to also
appear in a nonidentity context, and add anti-identity hard negatives.

## What was added

`src/slm_training/harnesses/train_data/harness_qa.py` — reuses existing,
already-canonical machinery rather than inventing new parsing or scoring:

- **`AcceptedCompletionSetV1` / `build_accepted_completion_set` /
  `statement_prefix_boundaries`** — splits a canonical multi-statement
  document at a legal statement-line boundary (OpenUI documents serialize one
  top-level statement per line). The prefix is accepted as shown even when it
  contains a forward reference not yet defined in it (e.g. `root =
  Stack([hero, cta], ...)` before `hero`/`cta` are declared) — the real
  `@openuidev/lang-core` backend parses that without error, it just cannot
  resolve the reference from the prefix alone; `COMPLETE_SUFFIX` payloads are
  legal-shaped prefixes, not necessarily closed/reference-resolved documents.
  The accepted set starts with the canonical suffix and adds: verified
  surface variants of it via the existing, byte-for-byte reused
  `scope_corpus.decanonicalize_variants` round-trip check; and any
  caller-supplied `peer_suffixes` (completions observed elsewhere in the
  corpus for the same literal prefix) accepted only when `prefix + "\n" +
  peer_suffix` **round-trips byte-exact** through the parser. A plain
  "doesn't raise" check is not enough here: an *unresolved* reference is
  silently pruned by the real backend rather than rejected (e.g. an
  unreferenced multi-statement fragment collapses to just its first
  statement), so a weaker check would wrongly accept a peer whose content was
  actually discarded. `tests/test_harnesses/train_data/test_harness_qa.py::test_peer_suffix_that_parses_but_silently_drops_content_is_rejected`
  pins this directly.
- **`accepted_completion_outputs` / `complete_suffix_record`** — persists the
  *full* accepted set through the schema's existing
  `ExampleRecord.accepted_outputs` field: `openui` carries the canonical
  completion, `accepted_outputs` carries every other verified member. This is
  exactly the field the existing scorer already reads
  (`score_output_targets`, `evals/task_scoreboard.py`) — no new mass/weight
  mechanism was introduced, so downstream scoring is membership-based by
  construction and never penalizes a verified alternative for not matching an
  arbitrary single gold.
- **`build_complete_suffix_task`** — the `COMPLETE_SUFFIX` `HarnessTaskV1`
  (`slm_training.dsl.harness_dsl`, reserved by DSH1-03/SLM-355 but not
  previously produced): the payload is the legal prefix itself (always a
  complete, independently valid fragment — `COMPLETE_SUFFIX` asks the model
  to complete a valid prefix, not to guess inside an incomplete one).
- **`cap_identity_records` / `identity_row_counts`** — enforces a hard,
  declared per-family (`payload_kind`, `grammar_category`) identity-row cap
  and marks every kept `IDENTITY` row `meta["diagnostic"] = True`; rows
  beyond the cap are dropped, never silently retained.
- **`atomic_form_coverage_violations`** — flags any lexical atomic form
  (`meta["scope_kind"] == "lexical"`, `meta["task"] == "identity"`) whose
  exact surface text never appears — as a substring of a prompt or target —
  in any non-identity record supplied. Checked against `scope_corpus`'s own
  output in isolation, this **fails** for a small fixture (its
  `scope_canonical_*`/repair/typed families don't happen to touch every
  lexical form); checked against `scope_corpus` output combined with this
  module's own `COMPLETE_SUFFIX` records, it passes — i.e. `COMPLETE_SUFFIX`
  is part of what closes this coverage gap, which is exactly DSH1-08's job.
- **`same_symbols_different_structure_negative`** — reuses the existing
  verified corruption oracle
  (`slm_training.data.corrupt.oracle.build_scoped_corruptions`) to produce a
  same-vocabulary, differently-structured decoy for one `IDENTITY` row,
  rather than inventing new mutation logic.
- **`matched_structure_permuted_markers_negative`** — a same-skeleton decoy
  with its declared runtime markers (`runtime_symbols_for_payload`) cyclically
  permuted within role, in one simultaneous regex pass (longest surface
  first, so a shorter marker is never matched inside a longer one it is a
  substring of, and no marker chains into another's target). A cyclic
  permutation among same-role markers of different surface lengths does not
  preserve total byte length (documented directly in the docstring after an
  initial version wrongly assumed byte-length invariance and a test caught
  it) — only the statement/skeleton shape and marker identity are what
  change.

## Acceptance criteria: how this increment satisfies them

- **Identity rows obey the declared cap.** `cap_identity_records` drops every
  row beyond `cap` per `(payload_kind, grammar_category)` family
  (`test_identity_rows_obey_the_declared_cap`,
  `test_cap_drops_rows_beyond_the_declared_cap_never_silently_keeps_them`).
- **Every atomic production appears in a nonidentity task.**
  `atomic_form_coverage_violations` returns empty for the combined
  `scope_corpus` + `COMPLETE_SUFFIX` corpus
  (`test_every_lexical_atomic_form_also_appears_nonidentity`); the negative
  control (`test_coverage_violation_detected_when_a_form_is_identity_only`)
  confirms the checker actually detects a real gap rather than trivially
  passing.
- **All accepted completions replay to verified terminals.**
  `test_complete_suffix_target_replays_to_a_verified_terminal` re-validates
  every `output_targets` member (canonical and alternates) concatenated with
  the exact prefix the model was shown.
- **No arbitrary single-gold label is used where several completions are
  accepted.** `accepted_completion_outputs` / `complete_suffix_record` never
  drop a verified member; `test_accepted_completion_outputs_excludes_only_the_canonical_member`
  and `test_legally_continuing_peer_suffix_is_accepted` cover this directly.

## Verification and claim limits

This sandbox initially fell back to the pure-Python Lark backend because the
`@openuidev/lang-core` Node bridge's `node_modules` weren't installed
(`src/apps/openui_bridge/` had no `npm install` run yet — `bridge_available()`
false, `OpenUIHybridBackend` silently falls back to Lark). That fallback does
**not** perform the real backend's unreferenced-binding pruning or carry its
full component/enum schema, so it gave two rounds of false local confidence:
an under-referenced first-draft fixture (three independent, non-cross-
referencing statements) kept all three statements locally but the real
backend correctly collapsed it to one on push, and a peer-suffix check that
only tested "does this parse" (not "does it round-trip byte-exact") looked
correct locally for the same reason. Running `npm install` in
`src/apps/openui_bridge/` (`node`/`npm` are present in this container; just
not yet wired for this pack) fixed both classes of false-confidence at the
source instead of working around them: with the bridge available,
`tests/test_harnesses/train_data/test_scope_corpus.py` — previously seen
failing with `symbolic_surface_policy/v1 rejected staged target:
open_string:'"column"'` — passes cleanly, confirming that failure actually
was an environment gap, not a real code issue, and giving every new test
below a faithful, non-fallback run:

- `pytest tests/test_harnesses/train_data/test_harness_qa.py -q` → 22 passed.
- `pytest tests/test_harnesses/train_data/ -q` → 115 passed, 3 pre-existing
  failures in `test_staged_materialization.py`, all
  `generator version mismatch for 'pack.corpus_generator': plan='v18',
  active='v21'` — reproduced identically checking out the pre-this-PR base
  commit (`34c70de1`) into a clean worktree with an unmodified `versions.json`
  (`active='v19'` there, still `!= plan='v18'`): a stale hardcoded plan
  fixture, unrelated to and predating this PR.
- `ruff check` on both added files passes.
- `python -m scripts.verify_version_stamps --check --staged` is clean (bumps
  `harness.train_data` to v21 — `harness_qa.py` lives under that component's
  already-registered `src/slm_training/harnesses/train_data/` directory
  prefix, so no new component was needed; v20 was this same change's first
  draft, v21 is the peer-suffix round-trip fix plus the corrected
  cross-referencing test fixture, both on this same PR before merge).

No corpus build, train, model evaluation, benchmark, checkpoint, AgentEvals
publication, capability certificate, or ship claim was produced.
`COMPLETE_SUFFIX` payloads use the document-level statement-boundary split
only; a tokenizer/decode-time `CompletionForest`-backed prefix split
(matching DSH1-07's `PartialKind.PREFIX`) is follow-up work, not attempted
here.

## Reopen conditions

- If a future producer needs `COMPLETE_SUFFIX` prefixes that are *not*
  themselves complete top-level statement counts (e.g. mid-expression
  prefixes), this module's document-level split does not cover that — it
  would need the `CompletionForest`/`PartialKind.PREFIX` machinery from
  DSH1-07 instead.
- The `test_staged_materialization.py` stale-plan-version failure noted above
  should be filed and fixed separately (bump the fixture's hardcoded
  `pack.corpus_generator` plan version, or derive it from the live registry)
  — out of scope here since it predates and is unrelated to this change.

## Research lineage

Builds directly on DSH1-03 (`harness_dsl/v1`, SLM-355 — reserved the
`COMPLETE_SUFFIX` operation this increment now produces), DSH1-05
(SLM-357 — canonical preference relations; not depended on directly since its
branch is unmerged, but this increment's canonical/accepted-set split follows
the same spirit), and DSH1-07 (SLM-359 — partial-state classification; this
increment's document-level prefix/suffix split is a narrower, Torch-free
sibling of that work, not a dependency on its unmerged code). Not a paper
reproduction; it operationalizes the staged-harness requirement that CAP0
completion supervision be verified, multi-accepted, and bounded rather than
single-gold and unbounded.
