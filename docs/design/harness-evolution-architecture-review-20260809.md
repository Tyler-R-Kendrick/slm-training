# Harness-evolution architecture review: why the self-improvement loop cannot climb (2026-08-09)

**Status:** deep review of the harness/training architecture and the documented
eval record; no code changed. Findings are cited to file/line as of this
review's commit. Successor approaches are filed per I14 — every closed approach
below names its replacement.

**Scope:** the four review lanes were (1) the autoresearch/autotrain loop
control flow, (2) the documented experiment/eval record, (3) the formal and
mathematical machinery, (4) the train/eval harness and metric definitions.
Lane-by-lane detail is in the appendix; the body is the synthesis.

---

## 1. Verdict

The repo has built an unusually honest **admission-control system** — and no
**search system**. Preregistration, content-addressed campaign locks, Lean-4
certificates, fresh-seed confirmation, fail-closed `None` metrics, and the
version-stamp registry all work as designed. What they guard is a loop whose
experiments carry approximately zero bits of information each, selected by a
mechanism with no model of the hypothesis space, scored by a primary metric
that has no gradient toward the actual goal.

The documented record over 2026-07-14 → 2026-08-09:

| Measure | Value |
| --- | --- |
| Continuous autotrain cycles (`continuous_cycle_results/v1`) | 245 across 27 loops |
| Phase-A screening positives | 14 (5.7%) |
| Fresh-seed confirmations of those positives | **0 / 14** (100% rejected: `primary_quality_not_reheld`) |
| Decision-bearing result JSONs with `ship.eligible: true` | **0 / 613** |
| `ast_beq_rate` / `canonical_beq_rate` in every documented run | **0.0000** (no movement in 4 weeks) |
| E-series experiments documented | ~430 ids in `quality-experiment-matrix.md` |
| Durable promoted model improvements | **0** |

The fresh-seed confirmation gate is *working*: every screening "win" was seed
noise, and the gate caught all of them. That is the system honestly reporting
that its own screening tier produces no evidence. The failure is upstream: the
loop keeps purchasing measurements that are provably unable to decide anything,
then rotates to the next lever and buys another one.

This is not a tooling gap or an effort gap — 4,281 files under `docs/design/`,
`model.twotower` at v312, 261 versioned components. It is an architecture gap:
**every step is verified; the walk is not.** Nothing in the system models,
bounds, or optimizes the *sequence* of experiments, and nothing computes —
before spending a run — whether the run can possibly return an answer.

---

## 2. Root causes (ranked)

### RC1 — Screening measurements are below the decidability floor

Screening runs `n=3` smoke documents at 20–22 train steps under
`MAX_RUN_MINUTES = 3` (`src/slm_training/levers.py:20`,
`resources/experiments/autotrain_climb/policy.v1.json` `screening_smoke_n: 3`).
Consequences, all derivable *before any run executes*:

- `structural_similarity` and `meaningful_program_rate` move in quanta of
  ~1/3 per flipped document. The policy's `minimum_effect: 0.01` is **~33×
  below the measurement granularity**; every observed "effect" is one fixture
  document flipping. The 245-cycle record confirms it: metric values cluster
  at a handful of recurring attractors (0.0575 ×61, 0.1742 ×45, 0.3828 ×22)
  and the byte-identical delta `0.3267→0.3828` was independently
  "discovered" and then confirmation-rejected in **nine separate loops**.
- The sign test in `credit_engine._two_sided_p_from_signed_effects`
  (`autoresearch/credit_engine.py:351`) has a minimum attainable two-sided
  p of **0.25 at n=3**. No screening result can ever reject at α=0.05. This
  is arithmetic, not an empirical finding.
- The one fully rigorous campaign (locked power protocol,
  `iter-slm287-locked-power-protocol-20260724.md`: cluster bootstrap, exact
  McNemar, simulated MDE) measured a delta of exactly 0.000000 — rigor
  applied to evidence that carries no signal returns a rigorous zero.

The ship-gate floor (`default_min_suite_n: 20`, `rico_held: 1500`) already
encodes the right instinct, and `fixture_insufficient_n` is the single most
frequent blocker in the record (449 hits). But
`promotion_dispose.ignore_ship_insufficient_n_for_climb: true` instructs the
climb loop to disregard exactly that check at the one place it binds.

**Closed approach:** treating n=3/20-step screening deltas as decision inputs.
**Successor approach:** the power-feasibility gate of §4-R1 — an experiment is
inadmissible unless its minimum detectable effect at the declared budget is ≤
its preregistered plausible effect. Computable pre-run, fails closed.

### RC2 — The climb primary has no gradient toward the goal

The goal metrics — `ast_beq_rate`, `canonical_beq_rate`,
`binding_aware_meaningful_v2` — are the ones that distinguish "right program"
from "valid program". They are pinned at 0.0 in every documented run, so they
provide no gradient. The loop therefore climbs
`structural_similarity` (`harnesses/model_build/eval_runner.py:562`): a **regex
bag-of-component-names Jaccard (weight 0.7) plus a bracket-count "depth"
similarity (weight 0.3)**. It never abstains (hard 0.0 instead of `None`,
unlike every other metric), which is presumably why it became the primary — it
always produces a number. `tree_edit_similarity` (`:626`) is a literal alias
for it, consumed as if it were an independent signal.

So the loop optimizes a proxy that is satisfiable without semantic
correctness, while the metrics that measure semantic correctness sit on a 0/1
cliff the 1.6M-param/20-step models never touch. Classic reward-hacking
geometry: the whack-a-mole rejections in the record (structure ↑ while binder
F1 ↓, c1847; structure ↑ while loss diverges, c1855) are its signature.
Meanwhile `placeholder_fidelity` and `component_type_recall` are recall-only
(`eval_runner.py:487,631`) — the maximal emitter scores 1.0 — so the gate set
pushes "emit more" from two directions.

**Closed approach:** climbing bag-of-names similarity.
**Successor approach:** a graded semantic surrogate with proved consistency
(§4-R4): normalized tree-edit distance to gold on canonicalized ASTs — an
actual metric (symmetry + triangle inequality), continuous where `ast_beq` is
a cliff, and provably consistent with it (`d = 0 ⇔ ast_beq = 1`).

### RC3 — Selection has no model of the hypothesis space

The "hypothesis space" is `_SCREENING_ARM_BANK`: **55 hardcoded
`(slug, prose, knob-dict)` triples** in `scripts/run_autotrain_continuous.py:766`,
identified by a ~195-line ordered if/elif (`_arm_slug_from_knobs:4014`) where a
fall-through returns `None` and becomes invisible to all bookkeeping.
Selection is rotation by cycle index plus a hand-weighted linear score with no
uncertainty and no exploration term (`thrash_residuals.py:118`,
`soft_rank_slugs:372` — deterministic argmax, ties broken round-robin). And:

- `expected_information_gain` is a **free-text string field**
  (`autoresearch/schemas.py:955`, `min_length=8`); all 25 call sites are
  hardcoded English. `confidence` is a hand-typed literal, and 0.9 is a
  load-bearing threshold compared against those literals
  (`run_autotrain_continuous.py:8115`) — the loop hardcodes both sides of its
  own confidence test.
- The exhausted-knob ledger keys on an **exact hash of the knob dict**
  (`hillclimb.py:256`), while the matrix builder deliberately perturbs
  `seed = 100_000 + cycle` and `steps += cycle % 3` each cycle to *avoid
  knob-signature collision* (`:9533,9542`). The dead-approach memory is
  defeated by construction; the real protection is a hand-maintained
  slug-level skip set.
- Closure never generalizes: closing `literal-close` says nothing about the
  three sibling arms sharing the same `ltr_tail_loss_weight` lever. There is
  no lattice over lever subsets, no interaction model, no evidence
  inheritance from components to compositions.
- The bookkeeping is asymmetric in the wrong direction: a **single** (likely
  false, per RC1) positive wipes a slug's entire multi-seed null tally
  (`:4465`), reopening dead arms; nulls accumulate slowly, wins reset
  instantly.
- Prior-work discovery (`literature.py`) cannot alter what runs: no retrieved
  paper can add an arm to the bank — that requires a human editing a 12k-line
  driver. The categorical-novelty audit is filled from a constant template
  (`_matrix.novelty:9570`).

**Closed approach:** flat slug bank + rotation + exact-hash exhaustion.
**Successor approach:** typed lever lattice + posterior-based selection
(§4-R2/R3).

### RC4 — The loop is engineered never to stop, therefore never to conclude

Every hard stop has an automatic bypass: `THRASH_CAUSAL_CAP_RELAX` un-skips
capped families, `BANK_EXHAUST_PROMOTE_FALLBACK` converts exhaustion into a
promote attempt, `_synthesize_thrash_arms` mints new `compose-*` arms when the
bank empties, `CHAMPION_CONFIRM_FALLBACK` re-confirms an already-confirmed
champion just to have something to run
(`run_autotrain_continuous.py:10850-10931,1417`). Cadence, not evidence, gates
promotion attempts (`climb_policy.cycle_role_for_index`). The driver also
self-acknowledges its own `document` prerequisite
(`_self_heal_document_actions:2566`) — the iron-law evidence trail is partly
machine-generated prose vouching for itself. The repo's own closeout docs
state the bank is exhausted for this size class
(`continuous-openui-scheduled-0805a-c4-results.md`), and the driver's
priorities re-propose the same exhausted levers anyway.

**Closed approach:** non-terminating thrash with synthesized filler arms.
**Successor approach:** exhaustion as a first-class terminal verdict (§4-R7):
a bank-exhausted loop must emit a typed `regime_exhausted` conclusion naming
the binding constraint (budget, metric floor, capacity class) and halt until a
constraint changes — that verdict *is* the information the loop exists to
produce.

### RC5 — The formal layer proves steps, not the walk (and two backends are theater)

What Lean actually proves is real and valuable: interval-arithmetic soundness
for the metric-band stack language (`Band.lean:97` — an abstract-interpretation
containment theorem), Γ-filters only tighten, ranking cannot escape the legal
set, singleton ⇒ zero forwards, forest-verified drafts add nothing
(`ConstrainedDiffusion.lean`), all `sorry`/`axiom`-free with counterexamples
for the negative cases. But:

- The two required "independent prover backends" (`formal/structural.py`) are
  **single-instance tautology probes**: several laws are literally
  `return all(statuses) == all(statuses)` or `return not (... and False)`
  (`:106,130,159`), evaluated once on hardcoded defaults. Since
  `required_checkers=("python_structural","python_reference")` and
  `min_backends=2`, the formal loop can close with zero Lean involvement and
  two tautologies agreeing on one fixed input.
- `check_lean_kernel` (`formal/checkers.py:281`) runs `lake build` — package
  compiles — and never audits that the claimed theorem exists or matches the
  claimed statement. (The good pattern exists at `autoresearch/formal.py:518`,
  which audits `#print axioms` output; the checker just doesn't use it.)
- `efficiency_gain_lcb` (`harness_core/efficiency_gain.py:37`) returns
  `(mean, mean, mean)` at n=1 — a zero-width "lower confidence bound" that
  trivially passes `lcb ≥ 1`, wearing a statistical name while gating
  promotion. At n=2–5 it uses normal z=1.96 where Student-t (12.7 at n=2)
  is the defensible choice; fit error from the scaling-law inversion
  (α grid-searched in 0.05 steps, irreducible loss *guessed* as half the min
  observed, `scaling_fit.py:75-82`) is not propagated at all.
- Fixture-synthesized metrics are type-indistinguishable from measured ones:
  `slm165_interaction_factorial.py:388` derives `meaningful_program_rate`
  from `_hash_noise(...)` — deterministic SHA-256 pseudo-noise — in the same
  record shape as real scoreboards, separated only by a `claim_class` string.

**Closed approach:** counting tautology backends as independent verification;
point estimates labeled LCB.
**Successor approach:** §4-R6.

---

## 3. Audit against the requested mathematical program

The request: use reverse mathematics, compiler design, graph theory, topology,
category theory, lambda calculus, energy, etc. to *measure, validate, iterate,
and prove the computable ranges of possibilities* steering harness evolution.
Honest inventory of where each stands:

| Discipline | Genuinely present | Named but absent / misapplied | Highest-value missing application |
| --- | --- | --- | --- |
| **Reverse mathematics** (which assumptions suffice for which conclusion) | `MetricAuthority ∈ {theorem, assumption_backed}` (`Band.lean:226`); the 5-lane diagnosis matrix; `WitnessStatus`/`Coverage` tri-states that never let UNKNOWN license removal | Promotion claims don't carry a *minimal sufficient evidence base*; `ignore_ship_insufficient_n_for_climb` discards the one axiom (sample floor) the conclusion needs | An **evidence calculus**: every decision node declares the weakest evidence (n, seeds, suites, authority) sufficient for it, machine-checked at dispose time — the power gate of R1 is its first theorem |
| **Compiler design** | Real LALR(1) legality via interactive-parser state memoization (`fastpath/engine.py:63`); textbook nullable/productive/reachable fixpoints (`grammar_capabilities.py:150`); choice codec = derivation-space encoding; hash-consed `(parser, semantic)` state graph with path compression (`completion_kernel.py`) | Forced-emit is a hand-written 10-entry punctuation table (`_TERM_TO_TEXT`), sound but incomplete; `can_complete_with_holes` falls back to bracket counting; train-time grammar signal is a punctuation proxy for the decode-time automaton (`fastpath/losses.py:23`) | **Grammar × tokenizer product automaton** (the standard constrained-decoding construction, cited as arXiv:2508.10111 in a docstring but not implemented) — makes `is_deterministic_next` complete instead of merely sound, and makes forced-run/checkpoint claims exact |
| **Graph theory** | Bounded-BFS reachability with three honest verdicts and certified depth lower bounds (`slm299_edit_reachability.py`); incrementally-maintained transitive closure for cycle checks (`semantic_state.py:270`); Merkle DAG conversation store | The *experiment* space has no graph at all: 55 flat strings, no edges, no inheritance | **Evidence propagation on the lever-subset lattice**: arms as vertices of a partial order by knob inclusion; null closure propagates upward under declared monotonicity, component posteriors prior compositions (R2) |
| **Topology / metric geometry** | Interval containment proofs (`Interval.lean`); coverage-gated refinement monotone on the candidate powerset (`lattice_search.py:70`) | "Topology" in the codebase means UI-tree shape ops — a different word. No distance on program space: `structural_similarity` is not a metric (no triangle inequality, magic 0.7/0.3 weights, and the Lean `Nat` and ℚ versions are *different functions* via truncating division) | A **real metric on programs** (canonical-form tree edit distance): gives the search space neighborhoods and convergence, gives eval a graded signal (R4), and lets "computable range" statements be stated as balls, not point predicates |
| **Category theory / algebra** | `OpFamily` partition; per-edit `inverse_action` metadata; content-addressed identity everywhere | "CRDT event store" (`conversation.py:277`) has **no merge, no commutativity law** — it is git's Merkle-DAG model; the "edit algebra" checks 4 laws empirically on fixtures, never verifies `apply(inv(e), apply(e,x)) = x` | Make I11 true: define merge as a join on the event lattice and prove convergence (idempotent-commutative-associative), or re-scope the invariant; verify the edit groupoid laws in Lean over the bounded edit space (they're finite — decidable by enumeration) |
| **Lambda calculus / type theory** | `scope_env.py`: seven disjoint namespaces, shadowing/forward-ref policies, `StableSymbolId` ordinals = genuine α-invariance discipline | No typing judgment anywhere; slot contracts / binding / dataflow enforced by ad-hoc constraint stages in `compiler_draft.py` (3,006 lines) | A **small bidirectional type system** over binders + slot contracts: subsumes several `ConstraintStage` families into one checkable judgment, gives `UNSUPPORTED` verdicts a typing-derivation certificate, and is the natural CAP-ladder rung between grammar-2-AST and NL-2-AST |
| **Energy** | LeverProof bands are metric-generic and already list energy as a supported unit (`leverproof-integration.md`); lexicographic candidate policy includes energy | No energy is ever observed (no run records joules/W·s); no energy-based *model* — ranking is stupid-backoff n-gram; the one mention of a global energy critic is an explicit not-done (`slm160_spv_disposition.py:328`) | Two separate items: (a) record energy per run as a first-class `CostKey` axis beside params/time — the Band machinery needs zero changes; (b) an **EBM/contrastive global scorer** over complete programs as a preregistered candidate for the residual ranker — the first lever family that plausibly moves `ast_beq` rather than the bag-of-names proxy |
| **Statistics / information theory** (the actually-binding gap) | Fresh-seed confirmation; Holm step-down + sealed score ledger in `credit_engine` (real, but only on the promotion tier that is never reached); locked power protocol exists (used once) | EIG is prose; screening has no p-value/CI and cannot have one at n=3; LCB degenerate at n=1; normal z at n≤5 | **R1 + R3 below** — power-feasibility as an admission theorem, numeric EIG, posterior selection |

The pattern across all seven rows: the *decode/legality* side of the house has
real mathematics (automata, fixpoints, interval proofs, certified search); the
*experiment-selection* side has vocabulary. The invariants (I1–I15) constrain
what a model may emit; nothing constrains what the loop may spend.

---

## 4. Redesign: make the experiment space a certified object

The unifying principle, in the repo's own idiom: **treat the experiment loop
as a decode problem.** Decoding here already works exactly as the user asks —
a forward-calculated symbol table bounds the legal domain *before* the model
runs, singletons bypass inference, speculation verifies against the oracle
before commit, empty domains are dead ends rather than fallbacks (I1–I6). The
experiment loop should obey the same laws over its own domain:

> Compute the set of experiments that *can* return information under the
> declared budget before spending anything; commit forced conclusions without
> running; rank the remainder by expected information; verify every
> speculative win before it schedules more compute; and treat an empty legal
> domain as a terminal verdict, never as license to synthesize filler arms.

Concretely, in priority order:

**R1 — Power-feasibility gate (the "singleton bypass" of experimentation).**
Extend `ExperimentCampaignV1` with a required, machine-checked block:
`{budget, n, seeds, metric_granularity, mde_at_alpha, plausible_effect}` where
`mde_at_alpha` is *computed* (exact binomial/sign-test tables for quantized
metrics at small n — decidable arithmetic, a natural LeverProof addition since
it is Nat/ℚ interval math). Admission rule, fail-closed:
`mde_at_alpha ≤ plausible_effect` or the campaign is `infeasible_by_design`
and never executes. Retire `ignore_ship_insufficient_n_for_climb`. This one
gate would have refused ~all 245 recent cycles *before* they ran — which is
exactly the point: the loop's last month of output was computable in advance,
for free. Screening at n=3 may survive only as an explicitly non-decisional
wiring check whose numbers can never enter selection state.

**R2 — Typed lever lattice replacing the flat arm bank.** Levers become typed
axes (already 90% present in `levers.lever_catalog()`); an arm is a point in
the product space; the bank is generated, not hardcoded; `_arm_slug_from_knobs`
is deleted in favor of canonical arm identity = content hash of the *typed
projection* (excluding seed/steps jitter — fixing the RC3 self-defeat).
Evidence attaches to lattice elements: a multi-seed null on a lever subset
closes its compositions unless an interaction hypothesis is preregistered;
component posteriors form the prior for compositions. Exhaustion becomes a
*coverage* statement over the lattice ("all rank-1 arms closed at budget B")
— a computable range of possibilities, stated and checkable.

**R3 — EIG as a number; selection as posterior sampling.** Replace the
`expected_information_gain: str` field with a computed quantity (expected
reduction in posterior variance of the arm's effect, or expected bits on the
promote/reject decision) under a per-arm posterior maintained from all past
cycles (a normal or Beta posterior per lattice element suffices; conjugate
updates are exact rational arithmetic — Lean-checkable if desired). Select by
Thompson sampling or UCB over that posterior: exploration becomes principled
rather than round-robin, a single noisy win *shifts* a posterior instead of
erasing a null ledger, and the hardcoded 0.9-confidence handshake disappears.

**R4 — Give the objective a gradient.** Adopt normalized canonical-form tree
edit distance as the graded semantic primary; prove in Lean the cheap parts
(it is a metric on the quotient by canonicalization; `d=0 ⇔ canonical_beq`)
and keep the ship gates on the exact-equivalence rates unchanged. Fix the
known metric defects in the same change: precision-side gates beside the two
recall-only metrics, delete the `tree_edit_similarity` alias, make
`structural_similarity` abstain like every other metric, and reconcile the
Lean Nat-vs-ℚ divergence so "proved monotone" refers to the shipped function.
Promote `binding_aware_meaningful_v2` (thirteen minor versions, still
`candidate_pending_calibration`) through the version-stamp process or delete
its gate claim.

**R5 — Split the run budget by claim class.** `MAX_RUN_MINUTES=3` is correct
as a CI/wiring wall and wrong as a universal evidence wall — it is RC1's
binding constraint. Give evidence-bearing claim classes
(`promotion_candidate`, `ship_gate`) a separate, preregistered budget tier
(HF Jobs is the sanctioned surface for exactly this), and encode R1 so the
choice is forced: under the 3-minute wall almost nothing is admissible, and
the system says so rather than thrashing. The alternative — keeping one wall
and accepting that the loop can only ever do wiring checks — is also
coherent, but then the loop must stop *claiming* to climb (R7 verdict).

**R6 — Close the integrity gaps in the proof chain.** (a) Replace the two
tautology backends with property-based enumeration over the bounded domains
(the laws are decidable on the fixture universe) and make the Lean checker
audit theorem presence + `#print axioms` per claimed theorem, not
`lake build`. (b) Real small-sample intervals: Student-t or exact bootstrap,
n≥2 enforced for any field named `lcb`, fit-uncertainty propagated through
`invert_loss` or EG demoted from gate to diagnostic. (c) Make
measured-vs-synthesized a *type*: fixture synthesizers emit
`synthetic_metrics` under a schema that gate/selection loaders reject, so a
`_hash_noise` number can never sit in a key a decision reads. (d) Derive
forced-emit text from terminal regexes (or the R2 product automaton) instead
of `_TERM_TO_TEXT`, keeping the current table as a verified special case.

**R7 — Let the loop conclude.** Remove the four never-stop fallbacks; a cycle
whose legal experiment domain is empty emits a typed `regime_exhausted`
verdict naming the binding constraint and parks the loop until a constraint
changes (new lever family, new budget tier, new metric floor). Under I14 that
verdict closes approaches, not goals — it is precisely the "computable range
of possibilities" statement the user is asking the system to produce, and it
is the single highest-information output the current loop is architecturally
unable to emit.

Sequencing note: R1 and R3 are small (schema + arithmetic + a selection
function) and neutralize the waste immediately; R2 and R4 are the structural
investments; R5 is a policy decision the maintainer must make; R6 is hygiene
that restores trust in words like "proved" and "LCB"; R7 is what makes the
whole thing a *science* loop rather than a treadmill.

---

## 5. What must not change

For balance, the review also identifies load-bearing strengths that any
refactor must preserve: the VSS tri-state support oracle and its
certificate-checked monotone closure (`dsl/solver/`); `common_forced_run`'s
universally-quantified checkpoint argument (`runtime/decode_schedule.py:62`);
the `Band.lean` containment theorem and the Nat-observation certificate
pipeline; the fresh-seed confirmation gate (it is the component that kept a
month of noise out of the model card); fail-closed `None`/timeout metric
discipline; the version-stamp registry; campaign event-chain integrity; and
the asymmetric params-as-cost promotion check. Every one of these is rarer in
the wild than the machinery it guards.

---

## Appendix — lane findings (file:line evidence)

### A. Loop control flow (lane 1)

Two loops exist: a prose-only agent checklist
(`.agents/skills/autoresearch/references/loop.md`) and the executable driver
(`scripts/run_autotrain_continuous.py`, 12,214 lines; `run_cycle:10638`).
Cycle = git-sync gate → receipt gate → champion-queue reconciliation → intent
by fixed priority ladder (retry &gt; confirm &gt; cadence-promotion &gt; screening) →
arm from rotation + soft rank → programmatic `HypothesisMatrix`
(`_matrix:9498`) → content-locked campaign → execute → classify → handoff.
Feedback channels: `slug_stats.json` soft re-rank, residual boosts (fixed
constants), multi-seed null skip (`_recent_completed_nonpositive_slugs:4327`),
exact-hash exhausted-knob ledger (`hillclimb.py:307`), and prose priorities
recovered by suffix string-matching (`:8119`). No bandit, no posterior, no
computed EIG anywhere (grep confirms). The `CategoricalNoveltyAudit` is filled
from a constant template. Statistical rigor (Holm, sealed ledger, credit
recompute) binds only `promotion_candidate`/`ship_gate` classes; the screening
tier that makes all selection decisions runs `claim_class: diagnostic` with
point comparisons against hardcoded constants
(`_classify_metric_tradeoff:6738`).

### B. Documented record (lane 2)

Window 2026-07-14 → 2026-08-09. 245 `continuous_cycle_results/v1` files: 231
non-positive, 14 positive, 14/14 confirmation-rejected; 62 measurement-
incomplete. 613/613 decision JSONs `ship.eligible=false`. 1,063
`autotrain-wf-smoke-*` stubs from a 3-day burst (all `mpr=0.0`) inflate doc
counts ~40%. No champion registry exists; the only committed checkpoint
(`playground_demo/last.pt`) predates the current output contract and no
longer loads. All trained arms ≈1.6M params, 20–22 steps, CPU, smoke n=3.
Best-ever observed vs ship gates: `ast_beq` 0.0 (gate 0.20), `canonical_beq`
0.0 (gate 0.10), suites beyond smoke never run. Perf engineering did move
(P0→Q9 3.2×, playground ~9×) but is non-promotable for lack of a valid
quality anchor. Recurring failure taxonomy: `fixture_insufficient_n` (449),
`primary_metric_null_or_worse` (167), incomplete/timeout (83+60),
harness_failure (33), 100% fresh-seed collapse, exact-match zeros, metric
whack-a-mole, exhausted-lever re-proposal.

### C. Formal machinery (lane 3)

Two Lean projects, both CI-built, `sorry`/`admit`/`axiom`/`native_decide`-free
with a source scan *and* `#print axioms` audit in the LeverProof Makefile;
~180 + 21 theorems incl. `evalMetricProgram_sound` (Band.lean:97),
`gamma_only_tightens`, `rank_skips_illegal_higher_score`,
`singleton_forwards_optimum_zero`, `forest_verified_draft_subset`, with
explicit counterexample lemmas. Certificate pipeline recomputes and
structurally compares JSON via the compiled binary. Limits: `Ratio = Nat/Nat`
(no negatives → no signed deltas or log-likelihood bands; no gcd reduction →
denominator blowup); `structural_similarity` proved monotone but Nat and ℚ
versions differ (truncating division). Python gaps: tautology prover backends
(`formal/structural.py:106,130,159`), `check_lean_kernel` = `lake build` only
(`formal/checkers.py:326`), degenerate LCB (`efficiency_gain.py:46`),
scaling fit with guessed floor and gridded α (`scaling_fit.py:75-82`).
Genuinely principled non-Lean pieces: solver closure (`dsl/solver/closure.py`),
completion kernel interning + Kleene witness invariants
(`completion_kernel.py:54-142`), reachability three-verdict analyzer
(`slm299`), monotone refinement guard (`lattice_search.py:70`),
`select_smallest_sufficient` (`promotion_engine.py:84`). Naming without
structure: "CRDT" (no merge), "lattice" (subset order only), "edit algebra"
(fixture-checked, inverses unverified), no type system, no EBM, no program
metric, no grammar×tokenizer intersection.

### D. Harness architecture (lane 4)

Pipeline: `build_train_data` (strict gates, feedback artifacts) → `train_model`
(`ModelBuildConfig`: 395 fields; 249 CLI flags; ~365 fields with no registered
validity constraint) → `evaluate_suites` (5 suites) → `ship_gates`
(self-declared "Python preview", authority in AgentEvals) →
`promotion_engine`/climb policy. Models: TwoTower MaskGIT-style masked
discrete diffusion over DSL tokens (d=128, ~1.6M params scratch; SmolLM2-135M
context opt-in and flagged as the largest uncharged capacity jump;
`twotower.py` is 16,266 lines with ~40 aux heads) and GrammarDiffusion
trans-dimensional diffusion over a bounded typed production tree (d=96).
Eval→synthesis feedback is broken in both directions that matter:
`synthesis_feedback.json` is computed purely from build-time evidence; the
one eval→data channel (`feedback_adjusted_mixture`) consumes denoising-NLL and
*downweights* low-NLL families even when their generation quality is zero;
the nine typed `meaningful_program_v1_report` failure codes — the richest
diagnostic in the system — terminate in JSON that nothing consumes. Historical
scoreboard back-fill (`_slim_suite`, `ship_gates.py:167`) silently substitutes
`parse_rate` for `meaningful_program_rate`. Serial-only matrices
(`run_quality_matrix.py:2887`) under the 3-minute wall set the throughput
ceiling. Guard-vs-bypass table: every honesty guard has a documented escape
hatch, and the fixture/measured distinction survives only as a string.
