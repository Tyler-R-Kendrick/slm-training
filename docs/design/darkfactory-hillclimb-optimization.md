# Dark-factory hill-climb optimization (phase 1: deterministic memory + selection)

**Status:** implemented (phase 1); phases 2–3 specified below with wiring
pointers. Parent analysis:
[`harness-evolution-architecture-review-20260809.md`](harness-evolution-architecture-review-20260809.md)
(root causes RC1–RC5, redesign program R1–R7).

**Goal.** A hands-off factory: no human changes trajectory, unblocks cycles,
or authors hypotheses — the operator only checks status. Deterministic
machinery outranks LLM inference everywhere a deterministic answer exists
(the experiment-loop analogue of invariant I1), and the model roadmap layers
flow-matching / diffusion / energy techniques on top of the autoregressive
and masked-diffusion stack rather than replacing it.

## What phase 1 shipped

### 1. Cross-version evidence ledger (R2 substrate — durable memory)

`src/slm_training/autoresearch/evidence_ledger.py` +
`scripts/build_evidence_ledger.py` + committed artifact
`src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json`
(`autotrain_evidence_ledger/v1`).

The loop's live memory (`outputs/autoresearch/**`) is gitignored, capped at
120 deliveries, and lost with every container — while 1,500+ result JSONs
under `docs/design` were never machine-consumed. The miner deterministically
folds the two hill-climb schema families (the
`continuous_cycle_results/v1` name variants and
`autotrain_measured_result(s)/v1`) into per-arm sufficient statistics:

- **Join key** is the arm slug — the stable identity `_SCREENING_ARM_BANK`
  already uses — recovered from `arms[].experiment_id` and from candidate-id
  tokens inside `reasons[]` (the only arm identity the cycle records carry).
  Knob hashes are *not* used: the driver deliberately jitters
  `seed`/`steps` per cycle, so exact-hash identity never repeats (review
  RC3).
- **Eval partitioning:** observations are bucketed by the version-stamp
  components `(evals.scoring, evals.meaningful_program,
  harness.model_build.eval, gates.ship)`; cross-partition and unstamped
  history informs posteriors at `CROSS_PARTITION_WEIGHT = 0.5`, never
  silently pooled at full weight.
- **Deduplication** on `(campaign_id, cycle_index, slug, seed)`; sorted
  traversal makes rebuilds byte-identical.

First real build: 2,074 files scanned, 268 deduplicated observations, 26
arms. Rebuild with `python -m scripts.build_evidence_ledger --write`;
inspect with `--status`.

### 2. Posterior-UCB arm selection (R3 — evidence-driven, deterministic)

Per arm, an exact Normal-Inverse-Gamma conjugate update over the primary-
metric effect (`arm_posterior`), fed by committed-ledger stats merged with
the live loop's `slug_stats.json` via the parallel-axis fold. Selection
score is the deterministic upper confidence bound
`mean + exploration_c · sd(mean)` — no RNG, no LLM: identical inputs give
identical picks. Properties (tested):

- an unexplored arm carries prior-width optimism (`sd = prior_scale`), so
  exploration is principled rather than round-robin;
- repeated nulls sink an arm smoothly — one noisy win *shifts* a posterior
  instead of erasing a null ledger (fixes the RC3 asymmetry);
- residual boosts stay lexicographically dominant, exactly as in
  `soft_rank_slugs`, so timeout/residual steering is unchanged.

Wired at the head of `_select_recommended_slug`'s soft-rank block
(`scripts/run_autotrain_continuous.py`, `_evidence_ranked_slug`), gated by
climb policy `selection.mode == "posterior_ucb"` (policy v7). Any failure
falls open to the legacy soft rank; `selection.mode = "rotation"` restores
the old behavior wholesale. On the real ledger the selector's first picks
are the arms with the strongest observed effects
(`semantic-contrast-compiler-margin`, mean Δ +0.27 over 6 observations) —
signal the rotation selector never used.

### 3. Exact power floor on arm closure (R1, narrow form)

`power_gate` (policy v7) applies exact `fractions.Fraction` sign-test
arithmetic: an arm may be closed only after enough independent complete-null
cycles that the pooled sign test over `screening_smoke_n`-document cycles
*could have rejected* at `alpha`. With today's values (n=3/cycle,
alpha=1/20, `min_complete_null_seeds=2`) the floor is exactly satisfied —
2 cycles × 3 docs = 6 pairs, min two-sided p = 1/32 ≤ 1/20 — so current
behavior is unchanged and the gate now *enforces* that consistency against
future policy drift (e.g. dropping smoke n to 2 would automatically raise
required null cycles to 3). `power_feasibility_report(n, alpha)` gives the
typed pre-run feasibility statement for any proposed measurement.

### 4. Typed terminal verdict (R7, first step)

`AutotrainCycleHandoffV1.terminal_verdict` (optional, following the
`thrash_regime` field precedent) now carries a
`regime_exhausted_verdict/v1` payload when the compose synthesizer cannot
produce an untried size-matched arm: campaign identity, binding constraint,
the closed slug set, policy digest, and an explicit resume predicate. The
`repair_harness` action is preserved (handoffs require ≥1 action); the
verdict makes the terminal state machine-readable instead of a string
marker, so status surfaces and successor loops can react to *conclusions*,
not log grep.

### Versioning / compatibility

Climb policy `v6 → v7` (new optional `selection` + `power_gate` blocks;
`load_climb_policy` treats unknown blocks as optional).
`harness.autoresearch.experiment_campaign` `v195 → v196`. Known side
effect: the policy-file hash participates in `promote_authority_sha256`, so
previously `promoted`/`climb_accepted` champion-queue rows re-certify on the
next cycle — by design (authority changed).

## Operator surface (the "check up on it" contract)

- `python -m scripts.build_evidence_ledger --status` — per-arm evidence
  across all harness versions (obs / nulls / positives / mean effect).
- `cycle_handoff.json` → `terminal_verdict` — non-null means the loop has
  *concluded* a regime and names the constraint whose change resumes it.
- `docs/MODEL_CARD.md` + closeout docs — unchanged authorities for model
  state and ship gates.

## Phase 2 — flow / energy / diffusion arms (specified, next PR)

Deterministic-first still applies at decode: every technique below is a
*ranking* lever over already-proven-legal candidates (I5) — legality stays
with the grammar oracle, singleton bypass stays absolute (I2).

1. **Energy reranker over legal residual candidates** (shortest path,
   assets already exist): wire the orphaned `solver_energy_*` fields
   (`twotower.py:566-572`) through `ModelBuildConfig` → `factory.py`, add a
   `_solver_energy_bias` beside `_component_plan_bias` in
   `_select_compiler_path` (`twotower.py:10419/10698` — additive
   `weight · tanh(·)` bias over `candidate_ids` only), train with the
   existing `energy_regression_loss`/`energy_pairwise_loss`
   (`models/solver_energy.py`) against `rejected_mining` +
   `openui_hard_valid_v1` contrast pairs (never against `composite_reward`,
   which is the held-out judge). Fail-closed identity-order fallback and the
   permutation invariant come free from `CandidateEnergyRanker`. Register
   `energy_decode_weight` in `TRAINED_DECODE_REQUIREMENTS`; keep all six
   `CAPACITY_SCALING_LEVERS` at baseline and initialize via
   `isolated_aux_init` so the arm is size-matched; report the head's
   parameter delta.
2. **Global program-level energy critic** as a best-of-n selector:
   `global_semantic_critic.py` already defines the energy/value shape;
   train on hard-valid contrast + mined rejects, evaluate as a
   `_pick_best_of_n` alternative (`twotower.py:13173`).
3. **Flow-matching over legal edits:** reframe `LegalEditFlow`
   (`models/legal_edit_flow.py`, currently fixture-only and unreachable
   from `train_model`) as a twotower *lever* — a rate/hazard aux head over
   the exact legal-edit candidate batch — rather than a new `model_name`,
   since the whole arm bank and matched-control machinery assumes
   `model_name="twotower"`. Its `multi_positive_mass` objective is already
   the partition-style set loss.
4. **Trans-dimensional grammar diffusion:** `GrammarDiffusionModel` is
   selectable from `train_model` but dormant (one 200-step smoke). Add it
   as a preregistered *diagnostic control arm* on the grammar matrix, not
   the continuous bank, until it clears fixture parity with twotower.

Each becomes a new arm-bank entry with a zero-valued matched control
(`knobs()` base dict), enters the evidence ledger under its slug, and
competes under the same posterior selection — the factory then *decides for
itself* how much compute each technique deserves.

## Phase 3 — remaining review items

- Extend the ledger with per-`eval_key` staleness distance
  (`verify_version_stamps` `behind_by`) as a decay weight, replacing the
  flat `CROSS_PARTITION_WEIGHT`.
- Persist `sdlc_delivery.json` (the rich per-arm record) for every cycle in
  the closeout docs instead of the lossy `continuous_cycle_results/v1`
  shape, so future mining needs no reasons-string slug recovery.
- Campaign-lock `power_feasibility_report` into `ExperimentCampaignV1`
  (full R1 admission gate) once evidence-bearing claim classes get their
  own preregistered budget tier — a separate versioned change to the
  run-cap policy (`levers.py` `MAX_RUN_MINUTES` + `repo_policy
  --sync-run-policy`), never an ad-hoc exception.
- Fold the four never-stop fallbacks behind the terminal verdict so a
  `regime_exhausted` cycle parks the loop instead of synthesizing filler
  compose arms (R7 complete).
