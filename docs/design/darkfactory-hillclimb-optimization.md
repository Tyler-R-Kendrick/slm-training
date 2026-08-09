# Dark-factory hill-climb optimization (delivered)

**Status:** fully implemented — the memory/selection layer, the model
ranking levers, terminal governance, and ledger evolution all landed; the
completing delivery was executed from the committed one-shot specification
[`darkfactory-oneshot-implementation-prompt.md`](darkfactory-oneshot-implementation-prompt.md)
by three parallel work units (M/G/L). Parent analysis:
[`harness-evolution-architecture-review-20260809.md`](harness-evolution-architecture-review-20260809.md)
(root causes RC1–RC5, redesign program R1–R7).

**Goal.** A hands-off factory: no human changes trajectory, unblocks cycles,
or authors hypotheses — the operator only checks status. Deterministic
machinery outranks LLM inference everywhere a deterministic answer exists
(the experiment-loop analogue of invariant I1), and the model roadmap layers
flow-matching / diffusion / energy techniques on top of the autoregressive
and masked-diffusion stack rather than replacing it.

## Deterministic memory + selection layer

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
(`semantic-contrast-compiler-margin`, mean Δ +0.27 over 3 delta
observations, 6 records total) — signal the rotation selector never used.

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

## Energy + flow ranking levers (unit M — delivered)

Two size-matched, preregisterable screening arms now sit in the continuous
bank, both pure *ranking* levers over already-proven-legal candidates at the
compiler decision points (I5; singleton bypass untouched, I2):

- **`solver-energy-rerank`** — the previously orphaned `solver_energy` stack
  wired end to end: `solver_energy_decode_weight` (tri-state, checkpoint-
  preserving) through `ModelBuildConfig` → factory → `TwoTowerConfig`; a
  `CandidateEnergyScorer` head under its own checkpoint prefix via
  `isolated_aux_init`; pairwise energy training on cached gold compiler
  decisions (gold below siblings, margin 1.0) on the detached-auxiliary
  path; decode bias `weight · tanh(−energy)` with identity-order + counter
  fallback on non-finite raw energies, in both `_select_compiler_path`
  branches. Registered in `TRAINED_DECODE_REQUIREMENTS` and the
  compiler-path decode lever group.
- **`legal-edit-hazard`** — a flow-matching hazard head at the same seam:
  softplus rates over the legal candidate set trained with the
  `multi_positive_mass` set objective plus a size-normalized `total_hazard`
  regularizer (objectives adapted from `models/legal_edit_flow.py`);
  decode bias `weight · tanh(log softplus rate)` with the same fail-closed
  contract and counters.

Both arms carry `structural_aux_head_profile` values (`solver-energy`,
`legal-edit-hazard`) so the matched control prebuilds identical heads at
zero weights — parameter-count parity is tested. New `DecodeStats`
counters: `{solver_energy,legal_edit_hazard}_bias_{applications,choice_changes,fallbacks}`.
The arms compete under posterior-UCB selection like every other arm — the
factory decides from evidence how much compute each technique earns.

## Terminal governance (unit G — delivered)

- **Park + deterministic resume.** A bank-exhausted cycle emits the typed
  `regime_exhausted_verdict/v1` carrying a `bank_fingerprint` (sha256 of
  sorted bank slugs+knobs, climb-policy sha, `MAX_RUN_MINUTES`), persists it
  to `loops/<id>/terminal_verdict.json`, and writes loop state `BLOCKED`.
  `run_cycle` short-circuits with `REGIME_PARKED` while the fingerprint is
  unchanged and resumes (`REGIME_RESUMED reason=bank_identity_changed`,
  verdict archived) only when the bank identity actually changes — a new
  lever family, policy revision, or budget constant.
- **Filler fallbacks retired behind policy.** Climb policy v8
  `terminal.park_on_exhaust: true` disables compose-arm synthesis and
  confirm-seed burning on exhaustion; causal-cap relaxation and retryable
  promote heads stay (they are evidence-driven, not filler). Flag false
  preserves the legacy branching semantics.
- **Power admission on promotion.** Promote campaigns lock a
  `power_feasibility/v1` report (`ExperimentCampaignV1.power_feasibility`);
  dispose refuses a non-decisive report as `promotion_infeasible_by_design`.
  `measurement.promotion_suite_n: 6` pins the promotion suite at the exact
  sign-test floor (min two-sided p = 1/32 ≤ 1/20) so promotion stays
  decidable. Known side effect (intended): the policy-file sha participates
  in `promote_authority_sha256`, so queued champions re-certify.

## Ledger evolution (unit L — delivered)

- **Staleness-decayed pooling.** Stamped cross-partition buckets decay per
  version step behind the current eval key
  (`max(floor, CROSS_PARTITION_WEIGHT · decay^behind_by)`, defaults
  decay 0.9 / floor 0.1, overridable via the policy `selection` block),
  with the version distance read directly from `versions.json` history.
  Partially unknown stamped keys use the flat weight for the whole bucket;
  same-key buckets stay at 1.0; unstamped stay flat.
- **Rich delivery persistence.** Per-cycle closeout JSONs embed the full
  `sdlc_delivery.json` payload (`"delivery"`, schema
  `autotrain_sdlc_delivery/v1`), and the miner consumes both embedded and
  standalone delivery records — slug from `candidate_id`, seed from
  `arm_seed` — superseding reasons-string recovery without double counting.
- **Operator status.** `python -m scripts.build_evidence_ledger --status`
  now also prints the current eval key, the posterior-UCB top-10, and the
  screening power geometry.

## Remaining open item

The full R1 admission gate for *screening* (refusing statistically
undecidable screening cycles outright, rather than treating them as
advisory) stays coupled to a preregistered budget-tier change for the
run-cap policy (`levers.py` `MAX_RUN_MINUTES` + `scripts.repo_policy
--sync-run-policy`) — a separate versioned decision for the maintainer, not
an implementation gap: under the current 3-minute wall, screening cycles
remain evidence-tier `directional` and closure is already power-floored.
