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
  document at a legal statement-line boundary. Both the prefix and the
  canonical suffix independently re-validate through
  `slm_training.dsl.parser.validate` (confirmed empirically: OpenUI documents
  serialize one top-level statement per line, and any interior line-count
  split leaves both halves independently valid — no tokenizer or decode-time
  `CompletionForest` is needed for this document-level split). The accepted
  set starts with the canonical suffix and adds: verified surface variants of
  it via the existing, byte-for-byte reused
  `scope_corpus.decanonicalize_variants` round-trip check; and any
  caller-supplied `peer_suffixes` (completions observed elsewhere in the
  corpus for the same literal prefix) that independently validate against
  that exact prefix. A peer suffix that does not legally continue the prefix
  is dropped, never accepted unverified.
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
- **`matched_structure_permuted_markers_negative`** — a same-shape decoy with
  its declared runtime markers (`runtime_symbols_for_payload`) cyclically
  permuted within role, in one simultaneous regex pass (so no marker chains
  into another's target).

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

`pytest tests/test_harnesses/train_data/test_harness_qa.py -q` → 21 passed.
`ruff check` on both added files passes. `python -m scripts.verify_version_stamps --check`
is clean (bumps `harness.train_data` to v20 — `harness_qa.py` lives under
that component's already-registered `src/slm_training/harnesses/train_data/`
directory prefix, so no new component was needed).

This sandbox lacks `torch` and (apparently) the full `openui` DSL pack schema
authority (its native `lang-core`/`openui_langcore` bridge): the **existing**,
already-merged `tests/test_harnesses/train_data/test_scope_corpus.py` and
several `test_staged_materialization.py` cases independently fail here with
`symbolic_surface_policy/v1 rejected staged target: open_string:'"column"'`
and `DSL pack version mismatch` — reproduced with none of this change's diff
applied, so it is a pre-existing local-environment gap, not a regression from
this change (mirrors the same class of gap DSH1-07/SLM-359 documented). All
new fixtures in this change were built and independently verified against
placeholder-only surfaces (`":scope.field"`-style) that this sandbox's
degraded pack authority *does* admit, so the new tests are real, verified
runs — not skipped or fixture-mocked. No corpus build, train, model
evaluation, benchmark, checkpoint, AgentEvals publication, capability
certificate, or ship claim was produced. `COMPLETE_SUFFIX` payloads use the
document-level statement-boundary split only; a tokenizer/decode-time
`CompletionForest`-backed prefix split (matching DSH1-07's `PartialKind.PREFIX`)
is follow-up work, not attempted here.

## Reopen conditions

- If a future producer needs `COMPLETE_SUFFIX` prefixes that are *not*
  themselves complete top-level statement counts (e.g. mid-expression
  prefixes), this module's document-level split does not cover that — it
  would need the `CompletionForest`/`PartialKind.PREFIX` machinery from
  DSH1-07 instead.
- If this sandbox's `openui` pack schema authority gap above turns out to be
  a real regression (not an environment gap), the `docs/design/dsh1-03-*` and
  `dsh1-07-*` claim limits should be revisited too, since they hit the same
  symptom independently.

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
