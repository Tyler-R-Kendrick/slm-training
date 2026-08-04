# SLM-304 lever-varied experiment loop (2026-07-28)

Standing protocol (user instruction, 2026-07-28): lever-varied runs only — no
step-scaling, no capacity-scaling. After every run: append the result row to
the matrix below, postulate what went wrong / what worked / what to attempt
next, and mark any lever that violates the validity rules **invalid +
deactivated** (never re-run).

## Validity rules (what counts as an acceptable change)

1. **I6** — No lever that weakens constrained/grammar decoding
   (`levers.CONSTRAINT_WEAKENING_LEVERS`); weakening arms are diagnostic
   controls only, never candidates.
2. **VI** — No capacity growth as a quality lever. Arms must be size-matched
   (`levers.require_size_matched_arms`) or charge `EG_params`. Trainable
   params are reported per row.
3. **Honesty** — No silent gold/placeholder channels
   (`honest_slot_contract=True`); unconstrained arms are named controls and
   never feed promotion.
4. **Evidence** — A lever is only "worked" if it beats its size-matched
   control on a populated suite at identical steps/n/seed. insufficient_n rows
   are informative, never ship evidence.
5. **Deactivation** — A lever is deactivated when it (a) violates rules 1–3,
   (b) fails closed on its own contract after a genuine harness fix attempt,
   or (c) ties or loses to control twice with the same attribution. Deactivated
   levers are recorded here and not retried without a new, documented
   hypothesis (I14: approaches are disposable, goals are not).

## Corpora

- Train: `outputs/data/train/slm230_symbol_only_v3` (101 records, strict,
  contract-v2 opaque ordinals; supersedes v2 — v2's design_md carried a lone
  `:slot_4` from the old `default.DESIGN.md` prose example, breaking the
  slot-contract resolver; template fixed and corpora rebuilt).
- Eval: `outputs/data/eval/slm303_v3` (smoke 3, held_out 5, adversarial 4,
  ood 4, rico_held 0; all contract-clean). insufficient_n (<20/suite) — no
  ship claims possible from this suite; lever comparisons only.
- Eval v4 (built 2026-07-28, runs after E201 onward):
  `outputs/data/eval/slm303_v4` = v3 fixture suites + rico_held n=24 (local
  semantic_test.jsonl, 3 leakage-rejected, target 24 met). rico_held is now
  the only suite with gate-meaningful n (≥20).

## Result matrix

| Run | Lever (varied factor) | Steps | Params | smoke n=3 | held_out n=5 | adversarial n=4 | ood n=4 | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E200 | control (fixed-row DSL symbols) | 80 | 1,608,962 | mpr .333 / struct .174 / recall .250 | mpr 0 / struct .098 / recall .162 | struct .188 | mpr 0 / struct .130 | baseline; fails gates + insufficient_n |
| E203 | runtime_symbol_features=role_gated | 80 | 1,608,962 | mpr .333 / struct .174 / recall .250 | mpr 0 / struct .098 / recall .162 | struct .188 | mpr 0 / struct .130 | exact tie with E200 (tie 1/2 for family) |
| E204 | E203 + semantic candidate masks | 38 (wall-capped) | 1,608,962 | mpr .333 / struct .174 / recall .250 | mpr 0 / struct .098 / recall .162 | struct .188 | mpr 0 / struct .130 | exact tie with E200 (tie 2/2) — family deactivated |
| E262 | choice-sequence codec (output_tokenizer=choice) | 46 (wall-capped, cap 2.58 min) | ~control (d_model/layers unchanged; output-vocab delta only) | mpr 0 / struct .096 / recall .083 | mpr 0 / struct .057 / recall .067 | struct .076 | mpr 0 / struct .060 | regression vs E200 on every suite — codec family deactivated at this scale |
| E201 | symbol_slot_augmentation (slot permutation + alpha-renaming) | 36 (wall-capped) | 1,608,962 | mpr .333 / struct .174 / recall .250 | mpr 0 / struct .098 / recall .162 | mpr .5 / struct .188 / recall .458 | mpr 0 / struct .130 / recall .208 | exact tie with E200 (tie 1/2 for data-augmentation family) |
| E200-v4 | control re-run on slm303_v4 (rico_held n=24) | 41 (wall-capped) | 1,608,962 | mpr .333 / struct .174 / recall .250 | mpr 0 / struct .098 / recall .162 | mpr .5 / struct .188 / recall .458 | mpr 0 / struct .130 / recall .208 | **rico_held n=24: mpr .917 / struct .069 / recall .486** — first gate-meaningful suite; validity solved, content selection is the bottleneck |
| E205 | constraint_graph_mode=hybrid (on role_gated+masks stack) | 32 (wall-capped) | 1,608,962 | identical to E200-v4 | identical | identical | identical | exact tie incl. rico_held (.917/.069/.486) — decode-scheduling family tie 1/2; strengthens scaffold-bound-decode finding |
| MIX1 | raw RICO mixture (slm230_rico_v4, 122 rec = 101 fixture + 21 rico) | 40 (wall-capped; baseline-matched) | 1,608,962 | mpr 0 / struct .096 | mpr 0 / struct .057 | mpr 0 / struct .076 | mpr 0 / struct .060 | **regression — total output collapse**; rico_held mpr .917→0, recall .486→0; raw-mixture approach rejected |
| MIX2 | diluted RICO mixture (slm230_rico_d6, 103 rec, ~2% rico) | 45 (wall-capped) | 1,608,962 | mpr 0 / struct .323 / recall 0 | mpr 0 / struct .247 / recall 0 | mpr 0 / struct .296 | mpr 0 / struct .236 | **collapse again** (different attractor); rico_held mpr 0 / struct .098 / recall 0 — mixture family rejected (2/2) |
| E277 | ASAp distribution-aware MaskGIT decode (eval-only, E201 parent) | n/a (eval-only) | 1,608,962 | mpr .333 / struct .174 | mpr 0 / struct .098 | mpr .5 / struct .188 | mpr 0 / struct .130 | exact tie incl. rico_held (.917/.069/.486) — decode-ranking tie 1/2; exposed constant-output artifact (see run log) |
| E257 | bind_encoding=relative (De Bruijn binder refs) | 39 (wall-capped) | 1,608,962 | mpr 0 / struct .096 / recall .083 | mpr 0 / struct .057 / recall .067 | (not reached) | (not reached) | regression — collapsed (constant `root = Separator()`); eval timed out after 2/5 suites (~23 min/suite); partial, directional only per run-cap law |
| E257-SM | E257 step-matched (train_model resume segments, same entry point) | 79/80 | 1,608,962 | mpr .333 / struct .174 / recall .250 (identical to baseline) | — | — | — | **parity** — rico_held mpr .708 / struct .057 / recall .382 (baseline .917/.069/.486); earlier collapse was an under-training artifact, not the representation |
| E255 | B4 scratch control (grammar_ltr_primary=False base) | 39 (wall-capped) | 1,608,962 | mpr 0 / struct .096 / recall .083 | mpr 0 / struct .057 / recall .067 | mpr 0 / struct .076 | mpr 0 / struct .060 | degenerate floor at capped steps; rico_held not reached (eval timeout); partial — its only valid comparison is the matched E280 arm at the same budget |
| S1 | **capacity reduction** d_model 128→64, denoiser 4→2, context 2→1 (user-directed, declared experiment) | 80 (2 segments) | **297,994** (vs 1,684,547; 5.7x cut) | mpr .333 / struct .174 / recall .250 — identical | — | — | — | **parity at 5.7x fewer params** — rico_held .917/.069/.486 identical; capacity never the constraint; ~2x step rate; eval wall unchanged (decode-bound, not forward-bound) |
| E280 | macro_tokens=True (C3, matched vs E255 at same budget) | 44 (wall-capped) | 1,608,962 | mpr 0 / struct .096 / recall .083 | (not reached) | (not reached) | (not reached) | ties E255 floor at matched budget — macro-token family closed, no escalation |
| S2 | S1 + batch 4→16, lr 3e-4→1e-3 | 33 (wall-capped; 528 records seen vs S1 320) | 297,994 | mpr .333 / struct .174 / recall .250 — identical | — | — | — | identical again — batch/LR retune is metric-neutral in the saturated regime; train-side knobs exhausted |

mpr = meaningful_program_rate, struct = structural_similarity,
recall = component_type_recall. rico_held n=0 throughout (suite empty).

## Deactivated levers

- **runtime_symbol_features family (E202 surface / E203 role_gated / E204
  semantic masks), 2026-07-28** — deactivated per rule 5c: two exact ties
  (E203, E204) against the size-matched E200 control at identical steps/n/seed.
  Not load-bearing at 80-step/101-record scale. Re-admission requires a new
  documented hypothesis (e.g. a regime where feature shaping is first-order,
  such as corpus ≥1k records and n≥20 suites).
- **choice-sequence codec family (E262 output_tokenizer=choice), 2026-07-28** —
  deactivated: regression against control on every populated suite (smoke mpr
  0 vs .333; struct .096 vs .174; held_out struct .057 vs .098; ood struct
  .060 vs .130) with no compensating win. Caveat recorded honestly: the arm
  trained only 46/80 steps inside the matrix per-arm wall cap (2.58 min;
  codec steps are ~1.7x slower), so the comparison is not step-matched — but
  the gap is far larger than plausible step-count noise, and the codec also
  ~doubles eval wall time. Re-admission requires a step-matched rerun showing
  parity-plus at equal steps AND a wall-time budget that makes the codec arm
  affordable at scale.
- **v10 exact-state local-preference family (E249-E254, E263/E264),
  2026-07-28** — deactivated at this scale from the canonical ledger, without
  burning new runs: E249 and E252 already measured and rejected (semantic
  quality regressed/collapsed in both); E252-E254 fail-closed absent
  counterfactual set-valued evidence; E250/E251 consume the same
  legality-shadow events. Re-admission requires the documented successor: a
  counterfactual semantic decision-events corpus (I14 successor approach,
  open since 2026-07-16).
- **raw/diluted RICO train-mixture family (MIX1 17% dose, MIX2 2% dose),
  2026-07-28** — rejected: both arms collapse to a single constant output on
  every suite (mpr 0, recall 0) while fixture-only controls never collapse.
  Not dose-dependent across 2%-17%. Re-admission requires rico entering via
  a two-phase curriculum or complexity-preserving synthesizers with a
  demonstrated non-collapse smoke run first.
- **from-scratch representation-change family (E262 choice codec, E257
  relative bind), 2026-07-28** — deactivated at this scale: both collapse to
  a constant output (E262 full 5-suite evidence; E257 2/5-suite directional
  partial, eval cost prohibitive). Confounded by step mismatch (~half the
  control's steps inside the cap). Re-admission requires the registered
  protocol: step-matched training via capped `--resume` segments and/or
  warm-start adaptation from the fixture parent (freeze trunk, retrain
  representation-facing embeddings), so the arm measures the representation
  delta instead of re-rolling the collapse dice.

## Run log

### E200 control (2026-07-28)

- **What went wrong:** gates all fail at 80 steps; smoke mpr .333 vs .66,
  held_out/ood mpr 0.0. Suite n too small for gate evidence (<20). Earlier
  failures before this run: (1) rico_held suite rejected by marker contract
  (named markers) → rebuilt as slm303_v3; (2) v2 corpus design_md lone
  `:slot_4` (templatized prose example) broke the E200-family slot-contract
  resolver → fixed `default.DESIGN.md`, rebuilt train v3.
- **What worked:** full E200 pipeline executes end to end with real scores on
  populated contract-clean suites; size-matched baseline established
  (1,608,962 params).
- **Next:** E203 role-gated features arm at identical recipe; if it beats
  control on smoke/held_out recall, escalate E204 (adds semantic candidate
  masks); if it ties, E203 is a deactivation candidate per rule 5c.

### E203 role-gated features (2026-07-28)

- **Result:** metric-identical to E200 on every populated suite (same
  mpr/struct/recall to 6 decimals). The lever changed nothing at 80 steps.
- **What went wrong:** nothing mechanically — run executed cleanly. The null
  result says gating surface features off for binder roles is not load-bearing
  at this scale; 80-step training on 101 records is dominated by
  under-training, and feature gating can't move recall.
- **What worked:** clean size-matched comparison (same params, seed, n,
  steps). Tie #1 recorded for the runtime-symbol-features family (rule 5c:
  two ties → deactivate).
- **Postulate:** surface-feature shaping is a second-order lever; it cannot
  show signal while first-order constraints (steps, corpus size, suite n)
  dominate. E204's semantic candidate masks are a different mechanism (hard
  pruning) — if masks also tie, the whole family deactivates and effort moves
  to codec-level levers (E262) and suite expansion (rico_held population,
  n≥20) where evidence can actually register.
- **Next:** E204 (E203 + semantic candidate masks).

### E204 semantic candidate masks (2026-07-28)

- **Result:** exact tie with E200 again (identical metrics everywhere). Train
  wall-capped at 38/80 steps — scores tied regardless; the cap truncates
  training equally for these arms and does not change the verdict.
- **What went wrong:** nothing mechanically. Second null result for the
  family: hard candidate masks also move nothing at this scale.
- **What worked:** clean falsification. Two exact ties (E203, E204) against a
  size-matched control at identical recipe — the runtime-symbol-features
  family is now **deactivated** per rule 5c rather than left as zombie work.
- **Postulate:** at 101 records / 80 steps, model error is dominated by
  first-order limits (data volume, training extent, suite n), so no
  feature-shaping lever can register. Two directions carry signal: (1)
  codec-level levers (E262 choice-sequence codec — different representation,
  not a feature tweak); (2) suite expansion to n≥20 so gate-level comparisons
  become meaningful at all (rico_held population + fixture suite growth).
- **Next:** E262 choice codec (v12 family, registered/unrun).

### E262 choice-sequence codec (2026-07-28)

- **Result:** regression on every populated suite vs E200 (smoke mpr 0 vs
  .333, struct .096 vs .174; held_out struct .057 vs .098; adversarial struct
  .076 vs .188; ood struct .060 vs .130). First attempt wall-capped after
  train+smoke only (600 s background ceiling); retry at 2700 s completed.
- **What went wrong:** the pure grammar-choice output stream learns slower per
  wall-minute — only 46/80 steps inside the matrix per-arm cap (2.58 min)
  where the lexer control finished 80 — and even so scores far below control.
  Eval also ran ~2x slower (~6 min/suite). The codec removes the copy-friendly
  lexical surface the tiny model was exploiting; at 46 steps on 101 records it
  has nothing to replace it with. So the comparison is confounded on steps,
  but the size of the gap (mpr .333 → 0) is not a step-count artifact.
- **What worked:** mechanically clean run end to end (real scores on all four
  populated suites, AgentEvals assertions emitted, certified_fallback=0
  everywhere — grammar constraint held). The retry pattern (longer background
  budget) is now the template for slow arms.
- **Postulate:** representation-changing levers (tokenizer/codec) are
  first-order but *negative* at small scale: they destroy the model's
  cheapest shortcut (surface copying) before the grammar-choice signal can be
  learned. Such levers need either much larger training budgets or a warm
  start; at fixture scale they can only regress. Family deactivated (see
  Deactivated levers). Signal-carrying directions remaining at this scale:
  data-level augmentation (E201 alpha-shuffle — changes generalization
  pressure without changing representation) and suite expansion to n≥20.
- **Next:** E201 slot permutation / alpha-renaming augmentation (v8).

### E201 alpha-shuffle augmentation (2026-07-28)

- **Result:** exact tie with E200 on every populated suite (smoke/held_out/
  adversarial/ood all identical to 4+ decimals). Train wall-capped at 36/80
  steps (same cap pattern as E204). Lever verified active in feature_flags
  (`slm.symbol_slot_augmentation=True`); arm is flag-clean vs control — the
  only difference is the augmentation.
- **What went wrong:** nothing mechanically. The augmentation permutes slot
  order and alpha-renames slot symbols at train time, but eval targets keep
  canonical slot naming — at 36 steps on 101 records the model is so far from
  fitting even the canonical distribution that renamed variants add no
  generalization pressure it can use. Identical scores suggest the augmented
  variants are effectively noise dilution at this scale, not signal.
- **What worked:** cleanest arm so far (single-flag delta, verified active,
  size-matched, honest contract on). Fast run (~5.5 min total).
- **Postulate:** data-level augmentation is a regularizer — it pays off only
  when the model has capacity/steps to overfit the un-augmented distribution.
  At heavy under-training it dilutes scarce gradient signal instead. Tie 1/2
  recorded for the data-augmentation family; a second augmentation tie
  deactivates the family at this scale. The real blocker for every lever so
  far is identical: 101 records / <80 steps / tiny suites. Highest-value next
  move is not another lever on v3 — it is the v4 baseline (rico_held n=24)
  to get one suite where gate-level evidence can register at all.
- **Next:** E200 control re-run on `slm303_v4` (rico_held n=24 baseline).

### E200 control on v4 — rico_held n=24 baseline (2026-07-28)

- **Result:** first gate-meaningful suite in the loop. rico_held n=24:
  **mpr .917 / struct .069 / recall .486** (gates: mpr ≥.1 PASS, recall ≥.15
  PASS, struct ≥.2 FAIL). Fixture-suite metrics identical to the 80-step v3
  run despite this arm wall-capping at 41 steps.
- **What went wrong:** struct .069 on rico_held — the model emits valid,
  meaningful programs whose *content* is wrong (component recall .486 but
  near-zero structural overlap). Also notable: fixture metrics are
  step-insensitive (36/41/80 steps → identical scores to 4 decimals), which
  means decode at this scale is scaffold-bound (grammar fastpath/copy probes
  do the work; learned weights barely register). That single fact explains
  every tie in this loop so far — and why only the representation change
  (E262) moved scores at all, downward.
- **What worked:** the constraint stack does its job at scale: 22/24 rico
  screens produce meaningful programs with zero certified fallback. The
  I6/I2 machinery is validated on real RICO input, not just fixtures. And
  the n≥20 suite finally separates metrics that were mushed together at n≤5.
- **Postulate:** the ship blocker is now precisely located — **content
  selection inside the legal domain**, not legality. Levers that reshape
  features or data at fixture scale cannot move it (scaffold-bound decode +
  under-training). Directions that can: (1) decode-side ranking levers that
  change *which* legal symbol is picked (constraint-graph scheduling E205 —
  I4-aligned, registered in v8); (2) training-signal levers that target
  content accuracy directly. Step/capacity scaling remain off-limits per
  user directive.
- **Next:** E205 constraint-graph scheduling on v4.

### E205 constraint-graph scheduling (2026-07-28)

- **Result:** exact tie with E200-v4 on all five suites, rico_held included
  (.917/.069/.486 identical to 4 decimals). Train wall-capped at 32 steps.
- **What went wrong:** nothing mechanically. The hybrid constraint graph
  changes *scheduling* (when/where compute is placed), not *ranking* — and at
  this scale the decode outcome is fully determined by the scaffold, so a
  scheduling change has no metric surface to move. This is the second
  independent confirmation of the scaffold-bound-decode finding: across
  E201/E203/E204/E205, no lever that leaves representation + ranking
  untouched moves any metric.
- **What worked:** falsification is cheap and clean (~10 min arms). The
  pattern is now diagnostic rather than anecdotal: ties are the expected
  output for any lever that does not change content ranking.
- **Postulate:** to move struct on rico_held, a lever must alter *which legal
  symbol is chosen* at non-singleton branch points (ranking), not features,
  not schedules, not augmentation. Candidates among registered arms: the v10
  local-decision rows (E248-E254) — need to check their definitions for a
  ranking-delta arm. Representation rows (v11 E255-E257) risk repeating
  E262's negative result at this scale.
- **Next:** inspect v10 row definitions; run the first ranking-delta arm.

### v10 local-preference family — adjudicated from canonical ledger, not re-run (2026-07-28)

- **Finding:** `docs/design/quality-experiment-matrix.md` (V10 section)
  already adjudicates this family: E248 measured (matched control, 4 gates
  failed); **E249 measured and REJECTED** (lexical objective generalized,
  semantic quality regressed); **E252 measured and REJECTED** (local held-out
  margin improved, semantic quality collapsed); E252-E254 fail-closed because
  no counterfactual set-valued evidence corpus exists; E250/E251 unrun but
  consume the same constraint-shadow events that certify grammar *legality*,
  not semantic preference. Our launch attempt confirmed the machinery
  (requires `--parent` + mined `--decision-events`) before the ledger check.
- **Decision:** not re-run here. Re-measuring a rejected approach with the
  same evidence kind would violate I14 (a rejected experiment closes an
  approach). **Family deactivated at this scale.** Re-admission requires the
  named successor: a counterfactual set-valued decision-events corpus
  (semantic, not legality-shadow) — the documented blocker since 2026-07-16.
- **What this implies for the loop:** the ranking levers that remain are not
  in v10. The rico_held evidence (mpr .917 / struct .069 / recall .486) says
  the model picks legal but wrong content — and the 101-record fixture train
  corpus contains no RICO-like structure to select from. The lever matching
  the bottleneck is the **training-data mixture** (add RICO-derived train
  records, leakage-filtered against v4 eval), not another decode or
  preference arm. Data-mixture is a registered, size/steps-neutral lever.
- **Next:** check `build_train_data` RICO sources; build a RICO-augmented
  train corpus; rerun the E200 recipe against it on v4.

### MIX1 raw RICO mixture (2026-07-28)

- **Result:** severe regression. rico_held mpr .917→0, recall .486→0,
  struct .069→.016; fixture suites collapse to the degenerate floor
  (smoke mpr .333→0). All 24 rico_held predictions and all 3 smoke
  predictions are the *identical* trivial program
  `root = Separator("vertical")` — total mode collapse to one output
  regardless of prompt. parse_rate stays 1.0 (grammar constraint holds);
  failure kind is `no_placeholders` on every record.
- **Ruled out as causes (checked):** corpus inventory identical (no new /
  dropped component types); every record incl. all 21 rico carries
  `:slot_N` placeholders; rico targets same length distribution as fixture
  (mean 124 vs 135 chars); steps matched the v4 baseline (40 vs 41 — both
  wall-capped); rico only 21/122 records and 4/162 sampled batch draws.
  Train loss 113→12.5 at 40 steps (fixture-only arms reach similar loss
  without collapse, e.g. E201 loss 15.7 scored identical to baseline).
- **What went wrong:** adding real RICO structure, even at 17% of records,
  destabilizes this tiny model's training within the fixed wall budget —
  the mixture raises per-step optimization difficulty enough that the model
  falls into the degenerate fixed-point (shortest legal program for every
  prompt) instead of the slot-bearing regime the fixture-only arms reach.
  Mechanistically: the collapse attractor exists in both corpora (13
  Separator records in each), but only the mixture tips the model into it.
- **What worked:** the failure is *diagnosable* — identical predictions
  everywhere, failure kind uniform, parse still valid. And the eval stack
  correctly scores it zero rather than crediting valid syntax (honest
  gates hold).
- **Postulate:** the content-selection bottleneck cannot be attacked by
  naive data addition at fixed budget — the optimization is too fragile.
  Dose-response is the open question: if a 5% rico dose preserves baseline
  behavior, mixture proportion is itself the usable lever (curriculum /
  capped fraction); if even 5% collapses, the rico records must enter
  through the same synthesizers as fixtures (template/aug pathways), not
  raw. Raw-mixture approach **rejected**; I14 successor = dose-capped
  mixture (MIX2), then synthesizer-mediated rico if MIX2 also fails.
- **Next:** MIX2 — rico-limit 6 (~5% dose), same recipe.

### MIX2 diluted RICO mixture (2026-07-28)

- **Result:** collapse again at ~2% dose (103 records, only 2 rico admitted).
  All 24 rico_held predictions identical (new attractor
  `root = Stack([TextCallout("warning")]…)`), recall 0 everywhere, mpr 0
  everywhere. Struct values *rise* (smoke .323 vs .174) but that is an
  artifact: a constant single output can coincide with more tree mass than a
  varied-but-wrong output; with mpr/recall 0 it is still total failure.
- **What went wrong:** dose-capping falsified. Two mixture arms, two doses
  (17%, 2%), two different collapse attractors. Six fixture-only runs in
  this loop never collapsed. Surface checks clear: rico prompts only
  modestly longer (132 vs 73 chars), same design_md size, same inventory,
  placeholders everywhere. Best remaining mechanism hypothesis: rico-derived
  records are structurally near-identical trivial programs (2-child stacks,
  mean target 124 chars) — they add probability mass to the trivial-program
  region, strengthening the collapse attractor the under-trained model
  falls into.
- **What worked:** cheap, decisive falsification of the whole "add raw-ish
  RICO to the train mix" direction at this scale (2 runs, ~25 min total).
- **Postulate:** train-mixture levers that include rico-pathway records are
  harmful at 101-record/40-step scale regardless of dose — **family
  rejected (2/2)**. I14 successor approaches, in order: (1) attack content
  selection with *fixture-synthesizer volume/complexity* instead (grow the
  corpus through layout_augment/template caps that preserve fixture style —
  the style that produced mpr .917); (2) two-phase curriculum (fit fixture,
  then blend rico) — deferred, schedule-adjacent and heavier. Direction (1)
  is next.
- **Next:** check build knobs for fixture augmentation volume; build a
  larger fixture-only corpus (target ~2x records) and rerun the E200 recipe.

### Fixture-volume probe → E277 decode-ranking arm (2026-07-28)

- **Fixture-volume finding:** the volume lever is nearly exhausted at current
  seeds — raising `--max-records-per-parent` 5→12 yields only 101→107
  records (synthesizers saturate; `--namespace-augment` is broken, stale
  `NamespaceAugmentSynthesizer` import at
  `harnesses/train_data/pipeline.py:1445` — harness bug, filed here, fix
  deferred out of loop scope). A 6% volume delta carries no signal, so the
  volume arm was not run.
- **Housekeeping:** the `qx_e200_symbol_control` run dir is reused by every
  E200-id arm; the MIX arms overwrote the v4 baseline checkpoint. Baseline
  metrics are preserved in this ledger; for eval-only arms the E201
  checkpoint (fixture v3 corpus, 36 steps, exact-baseline scores) serves as
  the parent. Matrix runner should arguably key run dirs by
  train-content-fingerprint — noted as harness improvement, not fixed here.
- **Next arm:** E277 ASAp distribution-aware constrained MaskGIT decode
  (eval-only decode-ranking lever, `--parent` E201 checkpoint) — directly
  targets *which legal symbol is picked*, the identified bottleneck, without
  touching constraints, size, or steps.

### E277 ASAp decode + the constant-output correction (2026-07-28)

- **Result:** exact tie with baseline on all five suites (rico_held
  .917/.069/.486 identical). Decode-ranking lever tie 1/2.
- **Major interpretive correction (honesty):** inspecting prediction
  diversity shows the earlier "validity solved" reading of the baseline was
  overstated. The fixture-trained model emits **one constant program for
  all 24 rico_held prompts** (`root = TextContent(":slot_1")`) and 1
  distinct output for all 5 held_out prompts; only the in-distribution
  smoke suite is varied (3/3). The baseline's rico_held mpr .917 is thus
  largely a *constant-output artifact* — a single slot-bearing trivial
  program passes the meaningfulness check. The only real content signal was
  and remains struct .069, and it says: **the model does not generalize
  content to held-out prompts at all** — it falls back to one safe
  slot-bearing attractor per prompt distribution.
- **What went wrong (loop-level):** every lever tried so far touches
  features, schedules, ranking, or data mixture — none can install content
  knowledge that is not in the weights. The weights lack it because the
  corpus is 101 small fixture records and the training budget is ~40
  wall-capped steps. This is why every in-scope lever ties or regresses.
- **What worked:** the tie itself is clean evidence (eval-only arm, same
  parent — no training confound), and the diversity check converts a
  misleading green metric (mpr .917) into an honest red one.
- **Postulate:** the ship blocker is *content generalization*, whose only
  remedies are training signal volume/diversity (data-capped, volume lever
  exhausted at current seeds) or training budget (user-capped). One untried
  registered lever class remains with a content-relevant mechanism: E257
  relative (De Bruijn) binder encoding — it changes how copy/structure
  references are represented, which could plausibly improve structural
  generalization without size/steps changes. Representation risk noted
  (E262 precedent). If E257 ties or regresses, the in-scope lever space at
  this scale is exhausted and the honest report is: the binding constraints
  are corpus diversity and training budget, both currently capped by
  policy/directive, not by any lever.
- **Next:** E257 relative bind encoding on v4.

### E257 relative (De Bruijn) bind encoding (2026-07-28)

- **Result:** regression, collapse. Smoke + held_out complete and identical
  to E262's degenerate signature (smoke mpr 0 / struct .096; held_out
  mpr 0 / struct .057; constant `root = Separator()` on every record).
  Background eval timed out after 2/5 suites — bind-relative eval runs
  ~23 min/suite on CPU, so adversarial/ood/rico_held were not reached.
  Recorded as directional partial, **not** gate evidence (run-cap law).
- **What went wrong:** second representation change, second collapse. The
  arm also confirms the comparison-validity problem raised in chat: bind
  steps are ~2x slower (39 steps in the cap vs 80 for control) AND eval is
  ~4x slower, so a from-scratch representation arm can neither train nor
  evaluate inside sane budgets at this scale.
- **What worked:** the pattern is now a rule with two independent instances
  (E262 codec, E257 bind): from-scratch representation changes collapse the
  under-trained model at this scale — full stop.
- **Postulate:** the representation family's failures are confounded by
  (a) collapse-attractor physics and (b) step mismatch; the current evidence
  cannot separate "bad representation" from "never trained enough". The
  registered re-admission protocol (below) addresses both: step-matched
  training via capped `--resume` segments (each ≤ MAX_RUN_MINUTES=3,
  composed to equal steps) and warm-start adaptation (init from the
  fixture-trained parent, freeze trunk, retrain representation-facing
  embeddings) so the arm measures the representation delta rather than
  re-rolling the collapse dice.
- **Decision:** representation family (E257, and re-confirming E262)
  **deactivated at this scale** pending that protocol.
- **Next:** build the step-matched resume protocol and re-run E257 to 80
  steps in capped segments.

### E257-SM step-matched relative bind (2026-07-28)

- **Protocol (new, registered for re-admission):** train via
  `scripts.train_model` from scratch in wall-capped segments, resuming from
  `checkpoints/last_full_state.pt` (`--resume-from` restores step/optimizer/
  RNG/pending batches; requires `last_full_state.pt`, not weights-only
  `last.pt`, and the *same entry point* — matrix-arm checkpoints are not
  optimizer-group-compatible with `train_model` resumes). Segments:
  25 → 51 → 79 of 80. Also learned: matrix `--resume` resumes run artifacts,
  NOT training — it silently restarts from step 0 (harness footgun, noted).
- **Result:** smoke mpr .333 / struct .174 / recall .250 — **identical to
  baseline**. rico_held mpr .708 / struct .057 / recall .382 vs baseline
  .917/.069/.486: same qualitative regime (constant output on OOD prompts —
  19/24 identical `TextContent(":slot_3","small-heavy")` + 5 empty), mildly
  worse. **The earlier E257 "collapse" was an under-training artifact.**
- **What went wrong:** nothing in the arm; the protocol cost is real
  (~8 min train segments + ~5 min eval for 2 suites vs ~3 min per matrix
  arm) — step-matching slow arms is ~4x the wall cost, which is exactly why
  the matrix cap produced the misleading collapse in the first place.
- **What worked:** the critique from chat is confirmed experimentally —
  representation arms were being judged at half the control's training. At
  matched steps, relative bind is *parity-class*, not harmful. This rescinds
  the strong form of the E262/E257 "representation is bad" conclusion:
  E262's gap (mpr .333→0 at 46 steps) is large enough that it may still be
  real, but it now needs the same step-matched re-test before the family
  verdict stands.
- **Postulate:** with under-training removed as a confound, relative bind
  shows no win — tie-class lever, not an improvement direction at this
  scale. The constant-output-on-OOD behavior is invariant across every
  lever and representation tried: it is a property of the 101-record
  corpus + 1.6M model + 80-step budget, not of any knob. Remaining untried
  registered rows: v16/v17 (unexamined), E256 HF pretrained denoiser
  (pretrained capacity change — conflicts with the no-size-growth
  directive; parked).
- **Next:** examine v16/v17 row definitions; also queue E262-SM (choice
  codec step-matched) to settle whether the codec gap is real.

### E255 B4 control (2026-07-28)

- **Result:** degenerate floor on all 4 completed suites (smoke/held_out/
  adversarial/ood all mpr 0, struct .057-.096); rico_held not reached —
  its `grammar_ltr_primary=False` decode path is ~4x slower eval and the
  run timed out. Partial, directional only (run-cap law).
- **Read:** the wall cap puts this base config at 39 steps → collapse
  regime, same as every slow arm. E255's numbers are meaningless against
  E200 but ARE the registered matched control for E280 at the same budget.
  The v16 matched-pair design is exactly the right protocol for this
  situation — comparing two equally-capped arms.
- **Next:** E280 macro tokens at the same budget; escalate to step-matched
  only if E280 beats the floor.

### S1 capacity reduction + E280 close-out (2026-07-28)

- **S1 (user-directed shrink):** d_model 64 / denoiser 2 / context 1 →
  **297,994 params (5.7x cut)** scores *bit-identical* headline metrics to
  the 1.68M baseline (smoke .333/.174/.250; rico_held .917/.069/.486).
  Full 80 steps in ~5 min (2 capped segments) vs never-fitting at 1.68M.
  This is the experimental proof of the invariant-VI thesis: capacity was
  never the binding constraint at this scale; the scaffold plus ~300k
  params saturates the 101-record corpus. Eval wall barely moved
  (12.1 s/record p50 at gen4/ltr128) — decode cost is per-token
  legal-domain computation, not forwards, so further eval speedups must
  come from decode-side flags (multitoken/bitsets/equiv cache — checkpoint
  config fields, need plumbing) and eval sharding, not model size.
- **E280:** ties the E255 floor at matched budget (smoke mpr 0, struct
  .096). Per the escalation rule (step-match only if it beats the floor),
  the macro-token family is closed without further spend.
- **Next:** S2 batch-16 + lr 1e-3 retune on the d64 base; then decode-side
  speed plumbing.

### Efficiency program (user-directed, 2026-07-28 evening)

- **S2 (batch 16, lr 1e-3 on d64):** identical metrics again (33 capped
  steps but 528 records seen vs S1's 320). Train-side knobs are exhausted —
  every train variation lands in the same scaffold-determined attractor.
- **Fast-decode config test (multitoken_accept + active_symbol_bitsets +
  equivalence_cache on the S1 checkpoint):** wall unchanged (5m04 vs 5m06
  for smoke+rico_held), metrics slightly *worse* (rico mpr .917→.792,
  recall .486→.424). Not adopted; flags reverted. Lesson: those flags
  optimize paths that are not the bottleneck.
- **Decode profile (1-record cProfile, 29.2s total, 12.1s in decode):**
  - ~8.9s/record (74%) in `build_completion_forest` — **72 forest builds
    per record** at ~0.12s each, each re-syncing an
    `OpenUIIncrementalEngine` from scratch (`engine._sync` 2174 calls,
    5.8s cumulative) via `dsl/pack.py:_tail_from` →
    `compiler_draft._build_openui_completion_forest`.
  - ~5.7s of that is one-time dynamic-import cost (lazy imports in
    `pack.py` hot functions) — amortizes across records, not the
    per-record sink.
  - `decode_stats` confirms: denoiser_ms=0, dfa_sync counted ~0 — model
    forwards are NOT the eval cost; the grammar forest rebuilds are.
- **The real eval-speed lever:** share one incremental engine across the
  72 forest builds per record (sync incrementally per prefix instead of
  rebuild-from-scratch), or cache forests keyed by (prefix, contract)
  across records. Estimated 3-4x eval speedup. Must preserve decode-proof
  semantics (I1/I2/I6 — fail-closed); needs its own tested change.
- **Adopted config going forward:** d64/L2/L1 (297,994 params) as the
  screening base — bit-identical quality, ~2x step rate, 80 steps fits one
  cap. Batch 16 + lr 1e-3 neutral; keep batch 4 lr 3e-4 (simpler).
  gen_steps 4 + ltr 128 for screening evals.

### Forest memoization attempt #1 — failed verification, reverted (2026-07-28)

- **What was tried:** process-local memoization of
  `compiler_draft._build_openui_completion_forest` keyed by
  (tokenizer id + strong ref, prefix, contract, bounds, flags), with
  `state` dropped from the key (it is an engine/prefix-text carrier).
- **Why it failed:** (1) hit rate only 8% (86/1098 builds) — decode extends
  the prefix every position, so exact-prefix keys almost never repeat;
  (2) **metrics shifted** (smoke struct .174→.123, rico mpr .917→.833) —
  stateful calls (pack.py:372, carries `request.state`) and stateless calls
  (pack.py:408 `_tail_from`) collide on the same key but are NOT equivalent
  (different engine sync paths). Failed bit-identical verification → fully
  reverted (`git checkout`). Lesson recorded: at this layer, reuse must be
  *structural* (the forest is a trie — after committing an action, the next
  domain is the subtree under that action, and a complete parent forest
  already certifies its paths' EOS reachability), not exact-key
  memoization. `GrammarDecodeState.completion_domain_cache` (models/
  grammar.py:217) already exists as a per-row cache slot — reuse target.
- **Next:** structural forest/trie reuse in the decode hot path with
  bit-identical verification as the acceptance gate.

### Structural forest/engine reuse — SHIPPED (2026-07-28, agent-implemented, parent-verified)

- **Mechanism:** `OpenUIIncrementalEngine.copy()` structural forks (shared
  parser/lexer) instead of fresh `__init__`+`set_prefix` per build;
  process-wide memo of Lark `accepts()` keyed by (grammar, LALR state
  stack); lazy accepts; `advance_checked` single-pass test-and-commit with
  suffix-lex fast path (plus a trim-safety fix); stateless forest builds
  memoized process-wide (stateful calls NEVER touch the cache — the exact
  collision that killed attempt #1); `_tail_from` EOS-witness proofs share
  one forest memo + one forked-engine lineage per domain computation.
- **Correctness gate:** bit-identical `prediction_sha256` /
  `raw_prediction_sha256` / `decode_outcome` on all 7 records under
  `--decode-timeout-seconds 300` (the only stable comparison — see below).
  14/14 new reuse tests pass; 21/21 DSL suites pass; version stamps bumped
  (`dsl.operators.registry` v6, `model.twotower` v262);
  `verify_version_stamps --check` + `repo_policy` clean.
- **Wall:** gated smoke+rico_held eval 5m06s → **4m01s** (verified by
  parent); no-timeout 7-record eval 4m05s → 1m18s (**~3x**).
- **Major discovery (changes loop interpretation):** the gated eval was
  wall-clock unstable BY DESIGN — every record sat on the 12s
  `decode_timeout_seconds` knife-edge, and three identical HEAD runs gave
  rico mpr .833/.875/.917. The baseline's constant
  `root = TextContent(":slot_0")` outputs were substantially **timeout
  fallbacks**, not model outputs. With decode now completing: rico_held
  struct **.069 → .209**, recall .486 → .604, 13/24 model_valid, 4
  distinct predictions (real content!). mpr "drops" to .667 only because
  the constant fallback trivially passed the meaningfulness check — the
  new outputs are real. Every earlier loop conclusion drawn from the
  gated-eval numbers (ties, .917 "validity solved", struct .069 floor)
  must be re-read through this lens: lever comparisons were partly
  comparing timeout behavior, not learned behavior.
- **Remaining headroom:** ~37 unique tail-proof builds per position at
  0.6-2s/position; next structural step is keying scoped completions by
  (LALR state stack + declaration-scope signature) instead of absolute
  prefix, and/or memoized min-EOS-length replacing the bounded witness
  search (the latter is NOT bit-identical by construction — changes
  `nodes_left=16` exhaustion behavior — so it needs its own gate).
- **Smoke caveat:** 2/3 smoke records still hit the 12s deadline in the
  gated eval (mpr 0 there); rico_held is the informative suite now.
