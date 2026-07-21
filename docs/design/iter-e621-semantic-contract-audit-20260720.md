# E621 — auditing `semantic_contract_for_openui` for E617/E618-shape bugs (clean)

Date: 2026-07-20
Status: completed, audit clean, no code change, no checkpoint trained

E620's Class-B audit (of `binding_aware_meaningful_v2()` in
`evals/meaningful_program.py`) explicitly flagged one thing as out of its
declared scope and unverified: `src/slm_training/data/quality.py`'s
`semantic_contract_for_openui`, which reimplements a regex-based mini-parser
(`_ASSIGNMENT_RE`, `_DECLARATION_COMPONENT_RE`, `_IDENTIFIER_RE`,
`_QUOTED_RE`) instead of using `dsl.parser.parse` — the same *shape* as
E618's bug. E620's own "next" list named the exact question to chase:
"confirm whether any current train-data builder still populates
`record.meta[\"semantic_contract\"]` with E613+-era typed-array/nested-object
syntax, and if so, whether its regex mini-parser misclassifies it." This
iteration answers that directly, and also re-checks `data/quality.py` more
broadly for an E617-shape gap (decode-time-consumed state populated only
behind an easily-unset flag).

## Grounding: how `semantic_contract_for_openui` is actually invoked

Traced every real caller (not just static reading):

- `src/slm_training/harnesses/experiments/teacher_paraphrase_activation.py:573`
  — `render_canonical_request` calls it, then `render_semantic_contract_prompt`
  turns the result into the literal GENERATE-mode training prompt text
  ("Component inventory: …; Declarations: …; Reference graph: …;
  Placeholders: …"). This is the most consequential call site: a bug here
  doesn't just reject a record, it can train the model on a false claim
  about its own target's reference graph.
- `src/slm_training/harnesses/train_data/pipeline.py:303` — attaches
  `record.meta["semantic_contract"]` at data-build time.
- `src/slm_training/data/edits/__init__.py:502` — `emit_transition_records`
  does the same for edit-derived GENERATE records, built from
  `transition.after`.
- `src/slm_training/data/quality.py:348` (`_semantic_contract_reasons`) —
  re-derives the contract from `record.openui` at admission-gate time and
  flags `semantic_contract_output_mismatch` if it drifts from the stored
  meta, which `independent_judge` folds into a record's admission reasons.

**Confirmed:** no caller in `twotower.py`, `choice_tokenizer.py`, or
`meaningful_program.py`. Unlike E617/E618/E620's targets, this function is
exclusively a data-build-time / admission-gate path — never a decode-time
bias, never an eval-time scorer of model output. That changes what "live"
means here: the inputs are always synthesizer-controlled OpenUI, not
adversarial/free model generation.

## E618-shape check: does the regex mini-parser misclassify valid syntax?

Three concrete failure modes were hypothesized and each checked against the
real grammar, the real schema, and the real on-disk corpus rather than left
as static reasoning:

**1. Multi-line truncation.** `_ASSIGNMENT_RE` has no `DOTALL` flag, so it
only captures a statement's first physical line. Built a hand-crafted
multi-line OpenUI example and confirmed `semantic_contract_for_openui` really
does truncate references to the first line when fed one — the theoretical
bug reproduces. But scanning the entire on-disk training corpus
(`src/slm_training/resources/data/train/*/records.jsonl`) confirms the real
canonical serializer output is always one physical line per statement (e.g.
`root = Stack([sep, cap])\nsep = Separator("horizontal", true)\ncap =
TextContent(":cap")`), matching every quoted example in
`docs/design/iter-e614/e615/e617*.md` (including the object-frame syntax:
`v0 = ImageGallery([{alt: ":ood.gallery.alt", src: ":ood.gallery.img",
details: ":ood.gallery.caption"}])` — one line). **Not reachable** via any
current builder.

**2. Object-key / binder-name collision.** Reference detection strips
quoted spans, then flags any bare identifier in what remains that also
equals a declared top-level binder name. This misfires if an object-literal
property key (e.g. `ImageGallery`'s `src`/`alt`/`details`) happens to
collide with a real binder name used elsewhere in the same program — a false
reference edge, independent of real semantics, exactly E618's bug shape one
level over. Checked from both directions:

- **Schema side:** scanned every `$defs` entry in the live schema
  (`lang_core.library_schema()`) for array properties with inline-object
  items. `ImageGallery.images` is the *only* hit; its item schema
  (`additionalProperties: false`) has exactly three keys: `src`, `alt`,
  `details`. No other component in the current grammar uses object-literal
  syntax at all.
- **Corpus side:** collected all 473 unique top-level binder names actually
  used across every `records.jsonl` in the training corpus. None is `src`,
  `alt`, or `details`.
- **Generator side:** traced the binder-name generator itself
  (`_TypedBuilder._binder`, `src/slm_training/data/progspec/generate.py:287-290`)
  — it stems names only from schema component-type names (the 54 keys in
  `openui_prop_order.json`'s `$defs`, none named `Src`/`Alt`/`Details`),
  never from property keys. No other synthesis path (teacher paraphrase,
  edits, frontier description) generates fresh binder names.

**Structurally prevented, not merely unobserved.** The collision can't occur
under the current grammar because the one object-literal component's key set
and the binder generator's name space are provably disjoint.

**3. Missed cross-statement references inside object literals.** The
`ImageGallery` item schema types `src`/`alt`/`details` as strings only (never
a `$ref`/identifier type), so their values are always quoted literals or
`:placeholder` tokens — never a reference to another top-level statement.
There is no reference to miss inside an object literal in the current
grammar.

## Closing E620's own "next" question directly

E620 asked whether any current builder populates `record.meta["semantic_contract"]`
*together with* E613+-era typed-array/nested-object output. Scanned the
entire on-disk training corpus for records where `meta.semantic_contract` is
a dict *and* `openui` contains an `ImageGallery` object-literal item
(`ImageGallery([{`, `{src`, or `{alt`): **zero matches**, across every
corpus directory. The one corpus that does carry populated `semantic_contract`
meta (`e297_semantic_contract_judge_v1`) always serializes `ImageGallery` as
the empty literal `ImageGallery([])` — it pre-dates E613/E614's typed-array-
item work. So even setting aside the structural-impossibility argument
above, the specific co-occurrence E620 asked about does not currently exist
anywhere in this repo's training data.

## E617-shape check

`data/quality.py` exposes a few helpers actually consumed at decode time:
`object_property_matches_slot_role`, `semantic_role_candidates`,
`slot_compatible_property_names`. Only `object_property_matches_slot_role`
is called from `twotower.py` (`_schema_role_slot_bias`, lines 4970-4972),
and that call site is already inside the branch gated on `weight <= 0.0 or
not frames or not slot_contract` (line 4920) — the same `self._slot_contracts`
gate E617 fixed and E620 re-verified every reader of. No new or additional
gate exists here; the rest of `data/quality.py`'s functions are pure,
stateless, and carry no conditionally-populated attributes of their own to
silently no-op on.

## Decision

No code change. This closes out E620's explicitly-scoped-out follow-up as
**audited and clean**, with concrete evidence at three levels (schema,
generator, corpus) rather than a static "looks fine" read — so a future
iteration doesn't need to re-open this exact question. No regression test
added: there is no reachable bug to guard against today, and the
structural-impossibility arguments rest on already-tested generated
artifacts (`openui_prop_order.json`, `progspec/generate.py`), not on new
logic this iteration owns.

Not a ship claim. No checkpoint trained, promoted, or synced. No previously
reported metric in this lineage changes.

## Next

1. If a future component adds object-literal typed-array items whose
   property-key names overlap common binder-name stems (i.e. a schema
   component-type name is reused as a property key elsewhere), re-run this
   collision check — it's sound today only because `ImageGallery`'s key set
   (`src`/`alt`/`details`) is disjoint from every component-type name in the
   schema.
2. If `semantic_contract_for_openui` is ever pointed at a multi-line
   pretty-printed OpenUI serialization (it currently never is), the
   single-line `_ASSIGNMENT_RE` truncation bug documented above would become
   live. Worth a `DOTALL`-aware rewrite or an explicit single-line
   precondition assertion at that point — not before, since adding one now
   would be speculative code with no exercised path.
3. This closes the "audit for E617/E618-class bugs" thread that E617/E618/
   E620 opened. Future iterations in this lineage should move to a
   different phase of the pipeline (see `.claude/skills/autotrain/SKILL.md`'s
   phase-routing table) rather than re-auditing the same two files again
   without new evidence to chase.

Evidence: [JSON](iter-e621-semantic-contract-audit-20260720.json).
