# One-shot implementation prompt: complete the dark-factory hill climb

**Audience:** LLM coding agents working in parallel, each in an isolated git
worktree of `Tyler-R-Kendrick/slm-training` at or after commit `26c38ce`
(merge of PR #1548). This prompt is self-contained: do not ask for more
context. Everything you need — mission, laws, file anchors, contracts,
acceptance checks — is here. Line numbers are anchors at `26c38ce`; verify
with `grep -n` before editing and trust names over numbers.

## Mission

Make the continuous autotrain loop a hands-off dark factory: it steers
itself from evidence, unblocks itself or concludes with a typed verdict, and
never needs a human to change trajectory or author hypotheses. Deterministic
machinery outranks learned or LLM inference everywhere a deterministic
answer exists. Flow-matching and energy-model techniques enter as
preregisterable *ranking levers* layered on the existing masked-diffusion +
autoregressive stack — never as replacements for grammar-constrained
decoding.

Already merged (do not redo): the cross-version evidence ledger
(`src/slm_training/autoresearch/evidence_ledger.py`, committed artifact
`src/slm_training/resources/experiments/autotrain_climb/evidence_ledger.v1.json`,
CLI `scripts/build_evidence_ledger.py` with `--write/--status/--check`),
posterior-UCB screening selection (climb policy v7 `selection` block), the
exact sign-test power floor on arm closure (`power_gate` block), and the
typed `RegimeExhaustedVerdictV1` on `AutotrainCycleHandoffV1.terminal_verdict`
(`src/slm_training/autoresearch/schemas.py:1036`).

This prompt completes the remaining work as three independent, parallel
work units — **M** (model levers), **G** (loop governance), **L** (ledger
evolution). Each unit is a complete deliverable on its own branch. An
integrator merges the branches, performs the single consolidated
`versions.json` bump, updates design docs, and opens the PR.

## Non-negotiable laws (violating any of these fails the unit)

1. **Constrained decoding is the product.** Levers may change how legal
   symbols are *ranked*, never whether output is legal (invariant I5/I6).
   New scorers add only additive biases or reorderings over
   already-proven-legal candidate ids. Never touch candidate membership,
   never add a fallback to full vocabulary, never register a new lever in
   `levers.CONSTRAINT_WEAKENING_LEVERS`.
2. **Singleton bypass is absolute (I2).** When the domain has exactly one
   legal candidate, no neural forward runs. `TwoTowerModel._select_compiler_path`
   (`src/slm_training/models/twotower.py:10243`) short-circuits singletons
   near `:10267` *before* the bias chain — leave that ordering intact.
3. **Capability is never bought with size.** Leave all six
   `levers.CAPACITY_SCALING_LEVERS` (`src/slm_training/levers.py:208`) at
   baseline. New aux heads are built only when their lever is on,
   initialized via `isolated_aux_init` (see `twotower.py:1789` for the
   component-plan example) so base-model RNG is unperturbed, kept out of the
   base state dict (own checkpoint key prefix), and size-matched against the
   control by the `structural_aux_head_profile` prebuild pattern (the arm
   and its zero-loss control build the *same* heads).
4. **Symbol-only output contract.** No feature, embedding, hash, or score
   may be derived from template-marker/placeholder *text* or user-defined
   names (`levers.PROHIBITED_TEMPLATE_SEMANTIC_LEVERS`, `levers.py:109`).
   Token ids, decision kinds, structural positions, and hidden states are
   fine.
5. **Trained-decode contract.** Every new `*_decode_weight` lever must be
   registered in `levers.TRAINED_DECode_REQUIREMENTS` (sic — grep
   `TRAINED_DECODE_REQUIREMENTS`, `levers.py:378`) so enabling decode
   without its owning training loss is a hard error, and in
   `LEVER_REQUIREMENTS` under the appropriate decode-path group (grep
   `_DUAL_PATH_DECODE_LEVERS` near `levers.py:343`).
6. **Fail closed, fail open correctly.** Scoring failures (NaN/inf/shape
   mismatch) degrade to identity order with a counter, never to an
   exception on the decode path and never to candidate removal — copy
   `CandidateEnergyRanker` (`src/slm_training/models/solver_energy.py:159`).
   Selection-upgrade failures fall open to the legacy path with a printed
   `*_WARN` line (see `_evidence_ranked_slug` in
   `scripts/run_autotrain_continuous.py` for the pattern).
7. **Hard run cap.** `levers.MAX_RUN_MINUTES = 3`. Do not launch training
   runs as evidence; your evidence is unit/fixture tests. Any command you
   run must finish inside the cap — use targeted pytest selections.
8. **No `versions.json` edits and no `docs/design` edits from work units.**
   The integrator owns the consolidated component bump and the design-doc
   update (this avoids three-way merge conflicts). Instead, END your report
   with the list of watched files you changed.
9. **External test cases.** If you add parametrized JSON-backed tests use
   the mirrored files under `src/slm_training/resources/test_cases/` +
   `python -m scripts.refresh_test_cases` — but plain pytest functions are
   preferred and sufficient here.
10. **Style.** Match surrounding code. Comments state constraints, not
    narration. Run `ruff check` on every file you touch.

## Environment (identical for every unit)

- You are in a git worktree sharing the main repo's object store. The main
  checkout with the installed virtualenv is `/home/user/slm-training`.
- **Do not run `uv sync`.** Run tests and lint with the main venv against
  your worktree:
  `cd <your-worktree> && /home/user/slm-training/.venv/bin/python -m pytest <targets> -q`
  and `/home/user/slm-training/.venv/bin/python -m ruff check <files>`.
  (pytest's rootdir config puts your worktree's `src` on the path.)
- Create and commit on the branch named in your unit
  (`git checkout -b <branch>`). Small, coherent commits; end each commit
  message with:

  ```
  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01Tx7RTWqSvZukyDS1j1A9qt
  ```

- **Report back** (final message): branch name; commit SHAs; files changed;
  new tests + their pass counts; the full pytest selections you ran and
  results; any deviation from this prompt with one-line justification; the
  watched-path list for the integrator's version bump.

---

## Unit M — energy + flow ranking levers (branch `claude/df-model-levers`)

Two new size-matched, preregisterable screening arms for the continuous
loop: an **energy reranker** and a **legal-edit hazard (flow) head**. Both
are additive biases over legal candidates at the compiler decision points of
the TwoTower masked-diffusion decode.

### Existing assets (use, don't reinvent)

- `src/slm_training/models/solver_energy.py` — complete, tested
  (`tests/test_models/test_solver_energy.py`), currently unwired:
  `CandidateEnergyScorer` (MLP over state ⊕ hole ⊕ candidate features,
  `:102`), fail-closed `CandidateEnergyRanker` (`:159`, permutation-only,
  `_multiset` guard, `fallback_count`), `energy_regression_loss` (`:228`),
  `energy_pairwise_loss` (`:242`, equal-cost pairs skipped).
- `TwoTowerConfig` already declares orphaned fields at `twotower.py:564-573`
  (`solver_energy_head`, `solver_ranker`, `solver_energy_hidden_dim`,
  `solver_energy_loss_weight`, `solver_energy_pairwise_weight`,
  `solver_energy_cost_version`, `solver_energy_fallback`) — nothing reads
  them. Wire them; add `solver_energy_decode_weight: float | None = None`
  (tri-state, `None` preserves checkpoint value — copy
  `component_plan_decode_weight` semantics).
- `src/slm_training/models/legal_edit_flow.py` — default-off rate model:
  `legal_edit_flow_losses` (`:118`) includes `multi_positive_mass`
  (−log Σp over the positive set — the partition-style set loss) and
  `total_hazard`. It has no `build_model` branch and must NOT become a
  `model_name`; reframe as a twotower lever (below).
- The end-to-end lever registration exemplar is **component-plan**; trace it
  before writing anything:
  `ModelBuildConfig` fields (`src/slm_training/harnesses/model_build/config.py:346-349`)
  → factory passthrough + runtime-override allowlist
  (`src/slm_training/harnesses/model_build/factory.py:572-583`, `:155`)
  → `TwoTowerConfig` (`twotower.py:405-410`) + `STRUCTURAL_AUX_HEAD_PROFILES`
  validation (`:136-145`, `:803`)
  → head construction via `isolated_aux_init` (`:1789-1801`)
  → checkpoint aux prefix list (`:2242`)
  → training loss gated on weight>0 using `gold_compiler_decisions` /
  `_compiler_decision_cache` (`:4250-4310`)
  → decode bias `_component_plan_bias` (`:6204`) with the deliberate
  `weight * tanh(...)` authority clamp (comment at `:6238-6244`)
  → bias applied in both `_select_compiler_path` branches (restricted
  `:10437` region, tree `:10698` region; there is a third call site near
  `:12190` — grep `_component_plan_bias(` for all)
  → lever groups in `levers.py:343-380`
  → arm bank entry (`scripts/run_autotrain_continuous.py:766`,
  component-plan at `:791-800`), knob identity keys `_LEVER_KNOB_KEYS`
  (`:694-751`), slug classifier `_arm_slug_from_knobs` (`:4014`, ordered
  if/elif — a fall-through returns `None` and the arm becomes invisible, so
  add explicit branches), matched-control zero base dict inside `_matrix`'s
  `knobs()` (`:9675` — add your levers' zero/off values there).

### M1 — energy reranker lever

- Fields: `solver_energy_loss_weight: float = 0.0`,
  `solver_energy_pairwise_weight: float = 0.0` (already in TwoTowerConfig),
  `solver_energy_decode_weight: float | None = None`,
  `solver_energy_hidden_dim: int = 64` — add the missing ones to
  `ModelBuildConfig`, pass through factory, honor in TwoTowerConfig.
- Head: instantiate `CandidateEnergyScorer(state_dim=d_model,
  hole_dim=d_model, candidate_dim=d_model, hidden_dim=solver_energy_hidden_dim)`
  via `isolated_aux_init` under checkpoint prefix `solver_energy_head.`.
  Features at a decision point: state = pooled context (same pooling the
  component-plan loss uses), hole = denoiser hidden state at the decision
  position, candidate = decoder token embedding of the candidate id. All
  are symbol-only-safe.
- Training: at each cached gold compiler decision point, the gold candidate
  must carry lower energy than the other *legal* candidates —
  `energy_pairwise_loss` over (state, hole) groups, weighted by
  `solver_energy_pairwise_weight`; optionally `energy_regression_loss` on a
  cheap cost proxy weighted by `solver_energy_loss_weight`. If a regression
  target is not cleanly available, make `solver_energy_loss_weight` own the
  pairwise objective and drop the separate pairwise knob from the arm —
  document the choice in your report. Never train on
  `harnesses/preference/composite_reward` (it is the held-out judge).
- Decode: `_solver_energy_bias(...)` returning
  `solver_energy_decode_weight * tanh(-energy)` per candidate (lower energy
  ⇒ higher bias), `None` when weight ≤ 0 or head absent; identity fallback
  + counter on non-finite output; wire into every `_select_compiler_path`
  bias-chain call site; add `DecodeStats` counters
  (`solver_energy_bias_applications`, `solver_energy_bias_choice_changes`,
  `solver_energy_fallbacks`) following the component-plan counter pattern.
- Registration: `TRAINED_DECODE_REQUIREMENTS["solver_energy_decode_weight"]
  = ("solver_energy_loss_weight",)`; add to the compiler-path decode lever
  group in `LEVER_REQUIREMENTS`.
- Arm: slug `solver-energy-rerank`, hypothesis
  `"A trained candidate-energy reranker over legal compiler decisions improves smoke structural_similarity without lowering parse_rate or binder_reference_f1."`,
  extras `{solver_energy_loss_weight: 1.0, solver_energy_decode_weight: 1.0,
  structural_aux_head_profile: "solver-energy", compiler_decode_mode: "tree"}`.
  Add `"solver-energy"` to `STRUCTURAL_AUX_HEAD_PROFILES` and make head
  construction respond to it so the size-matched control prebuilds the same
  head at zero weights (study how existing structural arms and the
  `treatment_key` precursor-control map near
  `run_autotrain_continuous.py:10121-10269` do this; mirror exactly).

### M2 — legal-edit hazard (flow) lever

- Fields: `legal_edit_hazard_loss_weight: float = 0.0`,
  `legal_edit_hazard_decode_weight: float | None = None` in both configs +
  factory.
- Head: small rate head (Linear `d_model → 1` applied per candidate over the
  same state⊕hole⊕candidate features, or Linear over hidden producing
  per-candidate rates via the candidate embeddings — pick the cheaper one
  and keep hidden dim fixed) under prefix `legal_edit_hazard_head.`, via
  `isolated_aux_init`, profile literal `"legal-edit-hazard"`.
- Training: adapt the flow objectives from `legal_edit_flow_losses` to the
  compiler-decision seam: `softplus` rates over the legal candidate set at
  each cached gold decision point; `multi_positive_mass` with the gold
  candidate as the positive set, plus the `total_hazard` regularizer against
  the legal-set size. Import from `models/legal_edit_flow.py` where shapes
  permit; otherwise implement local equivalents with a comment citing that
  module as the source of the objective.
- Decode bias: `legal_edit_hazard_decode_weight * tanh(log_softplus_rate)`
  (mirror `_component_plan_bias`'s bounded-rate treatment), same fail-closed
  rules and counters (`legal_edit_hazard_*`).
- Registration + arm: as M1, slug `legal-edit-hazard`, hypothesis
  `"A flow-matching hazard head over legal compiler decisions improves smoke structural_similarity without lowering parse_rate or binder_reference_f1."`.

### M acceptance

- New tests in `tests/test_models/` (extend `test_solver_energy.py`, new
  `test_legal_edit_hazard.py` or equivalent): head prebuild parity between
  arm and zero-weight control (identical parameter counts), bias applies
  only over supplied candidate ids, tanh clamp bounds, non-finite → identity
  order + counter, decode-weight-without-loss-weight raises via
  `require_valid_lever_configuration`, loss decreases over a few optimizer
  steps on a synthetic fixture.
- Driver tests: extend the slug-mapping tests in
  `tests/test_scripts/test_run_autotrain_continuous.py` (pattern:
  `test_new_successor_arm_slug_mapping`, `:905`) for both new slugs; keep
  `test_select_recommended_slug_rotates_and_skips` green (it pins rotation
  identities — if your bank insertion shifts them, update the pinned
  expectations *in the same commit* with a note).
- Suites that must pass: `tests/test_models/test_solver_energy.py`, your new
  tests, `tests/test_scripts/test_run_autotrain_continuous.py`,
  `tests/test_levers*` (grep the actual path), and
  `/home/user/slm-training/.venv/bin/python -m ruff check` on touched files.

---

## Unit G — terminal governance (branch `claude/df-governance`)

Make exhaustion a real conclusion: park the loop under a typed verdict with
a deterministic resume predicate, retire the filler fallbacks behind policy,
and gate decision-bearing campaigns on computed power feasibility.

### Anchors

- Driver: `scripts/run_autotrain_continuous.py` — `run_cycle:10724`;
  fallbacks `THRASH_CAUSAL_CAP_RELAX:10949`, `CHAMPION_CONFIRM_FALLBACK:10986`
  (via `_repeat_confirm_while_waiting_for_promotion`),
  `BANK_EXHAUST_PROMOTE_FALLBACK:11012`,
  `_self_heal_thrash_bank_exhaust:1480` (compose-arm synthesizer;
  `_BANK_EXHAUST_MSG:1245`); handoff writer `_write_cycle_handoff:8244`
  (terminal_verdict populated in its bank-exhaust `else` branch; loop-state
  write with `state="IDLE"` at `:8864`).
- Schemas: `src/slm_training/autoresearch/schemas.py` —
  `RegimeExhaustedVerdictV1:1036`, `AutotrainLoopStateV1:1134` (its `state`
  Literal already includes `"BLOCKED"`; check the `phase` Literal and extend
  it only if no existing value fits).
- Verdict builder: `evidence_ledger.build_regime_exhausted_verdict`
  (`src/slm_training/autoresearch/evidence_ledger.py`, end of file).
- Policy: `src/slm_training/resources/experiments/autotrain_climb/policy.v1.json`
  (v7; loader `src/slm_training/autoresearch/climb_policy.py:233` treats
  unknown top-level blocks as optional — same pattern as `selection` /
  `power_gate`). **Known side effect you must note in your report:** the
  policy file's sha participates in `promote_authority_sha256`
  (`climb_policy.py:206`), so editing it de-certifies queued champions —
  intended, by design.
- Campaign contract: `src/slm_training/autoresearch/experiment_campaign.py`
  — `ExperimentCampaignV1:133`, `validate_contract:193`.
- Power arithmetic (reuse, don't reimplement):
  `evidence_ledger.power_feasibility_report(n, alpha)`,
  `evidence_ledger.parse_alpha`.

### G1 — park + deterministic resume

- Extend `RegimeExhaustedVerdictV1` with `bank_fingerprint: str | None = None`
  and set it in `build_regime_exhausted_verdict` callers: fingerprint =
  sha256 of canonical JSON of (sorted bank slugs + their knob dicts from
  `_all_screening_arm_bank()`, climb-policy sha256, `MAX_RUN_MINUTES`). Add
  a pure helper in the driver (or evidence_ledger) to compute it.
- When `_write_cycle_handoff` emits a terminal verdict: persist it to
  `loops/<loop_id>/terminal_verdict.json` and write the loop state as
  `state="BLOCKED"` (choose/extend `phase` appropriately) instead of
  `IDLE`, with the verdict's binding constraint in the state payload.
- At `run_cycle` start (after git sync, before intent selection): if
  `loops/<loop_id>/terminal_verdict.json` exists, recompute the fingerprint.
  Unchanged → print a typed `REGIME_PARKED loop=<id> constraint=<...>` line
  and return a distinct cycle status without running any experiment (pick
  the pattern used for other early-return statuses in `run_cycle`).
  Changed → delete/archive the verdict file (rename to
  `terminal_verdict.resolved.<ts>.json` using an existing deterministic
  timestamp source in the driver, or the verdict's own cycle index), print
  `REGIME_RESUMED reason=bank_identity_changed`, restore state, proceed.
- The parked handoff still carries ≥1 action (schema requires it) — keep the
  existing `repair_harness` action.

### G2 — retire filler fallbacks behind policy

- New policy block (bump policy `version` to `v8`):

  ```json
  "terminal": {
    "park_on_exhaust": true,
    "description": "When the screening bank is exhausted, emit the typed regime_exhausted verdict and park the loop instead of synthesizing compose filler arms or burning confirm seeds. Causal-cap relaxation and retryable promote heads remain active (they are evidence-driven, not filler)."
  }
  ```

- With `park_on_exhaust` true: `_self_heal_thrash_bank_exhaust` must not
  synthesize compose arms (return `False` so the verdict path fires), and
  `CHAMPION_CONFIRM_FALLBACK` is skipped. `THRASH_CAUSAL_CAP_RELAX` and the
  retryable-promote-head branch of `BANK_EXHAUST_PROMOTE_FALLBACK` stay
  active in both modes. With the flag false, behavior is byte-identical to
  today.
- Existing driver tests that assert compose-synthesis / confirm-fallback
  behavior (grep `test_self_heal_thrash_bank_composes_successors`,
  `test_confirmed_champion_reconfirms_when_bank_exhausted_before_promotion`,
  `test_cycle_handoff_routes_exhausted_bank_to_model_build_repair`,
  `test_self_heal_cycle_error_recovers_bank_exhaust`): keep them passing by
  pointing them at a policy payload with `park_on_exhaust: false` (monkeypatch
  `load_climb_policy` or pass a policy object — follow how existing tests
  inject policy), and add new tests for the parked path (verdict file
  written, BLOCKED state, `REGIME_PARKED` early return, fingerprint-change
  resume).

### G3 — power admission on decision-bearing campaigns

- Add optional `power_feasibility: dict[str, Any] | None = None` to
  `ExperimentCampaignV1` with shape validation when present
  (`schema == "power_feasibility/v1"`).
- In the driver's promote-campaign construction (grep the promote matrix /
  `dispose_champion_promote` path), compute
  `power_feasibility_report(n=<promotion suite n from policy measurement>,
  alpha=parse_alpha(power_gate.alpha))` and embed it in the campaign before
  lock. At dispose time, a promote campaign whose report has
  `decisive: false` is typed as `promotion_infeasible_by_design` (a new
  reason string on the existing failure disposition — do not weaken any
  gate; this only *adds* a refusal). Screening campaigns are untouched
  (their evidence tier is already advisory).
- Tests: schema accepts/rejects shapes; promote dispose refuses a
  non-decisive report; feasible report passes through.

### G acceptance

`tests/test_autoresearch/` and `tests/test_scripts/test_run_autotrain_continuous.py`
fully green; new tests for G1/G2/G3; ruff clean. Report the policy-sha side
effect note.

---

## Unit L — ledger evolution (branch `claude/df-ledger`)

Deepen the factory's memory: staleness-decayed cross-version pooling, richer
per-cycle persistence for future mining, and a one-stop operator status.

### Anchors

- `src/slm_training/autoresearch/evidence_ledger.py`: `EVAL_KEY_COMPONENTS`,
  `CROSS_PARTITION_WEIGHT`, `_weighted_bucket_stats`, `current_eval_key`,
  `eval_key_from_stamp`, `build_ledger`, `extract_observations`.
- Version history order: `src/slm_training/resources/versions.json`
  (`components.<id>.history`, newest first) — read it directly with a small
  helper (do NOT import from `scripts/verify_version_stamps.py`;
  reimplement the ~10-line history-index lookup and cite it).
- Closeout docs writer: `scripts/run_autotrain_continuous.py`
  `_render_continuous_cycle_docs:2332` region and
  `_is_continuous_closeout_path` (grep) — this writes the per-cycle
  `docs/design/<campaign>-results.{md,json}`.
- Delivery payload: `sdlc_delivery.json` written per cycle under
  `outputs/autoresearch/<campaign>/` (gitignored) — the rich record with
  `candidate_id`, `arm_seed`, `arm_order`, `arm_exits`, `policy_sha256`,
  `direction`, `role`, metrics. Grep `sdlc_delivery` in the driver for the
  writer.

### L1 — staleness-decayed pooling

- Replace the flat cross-partition discount for *stamped* buckets with a
  per-component staleness decay: for a bucket whose key differs from the
  current eval key, compute `behind_by` = sum over the four
  `EVAL_KEY_COMPONENTS` of the history-index distance between the bucket's
  stamped version and the current version (0 when equal or unknown), and
  weight the bucket `max(floor, CROSS_PARTITION_WEIGHT * decay ** behind_by)`.
  `unstamped` buckets keep the flat `CROSS_PARTITION_WEIGHT`. Same-key
  buckets stay at 1.0.
- Policy knobs (read defensively from the `selection` block; do not edit the
  policy file — Unit G owns it; use defaults in code):
  `staleness_decay` default `0.9`, `staleness_floor` default `0.1`.
- Bucket keys are `"comp=ver|comp=ver..."` — parse with a small pure
  function; tolerate missing components.
- Pure, deterministic, unit-tested: equal key ⇒ 1.0; one-version-behind ⇒
  `0.5 * 0.9`; floor respected; unknown versions ⇒ flat weight.

### L2 — persist the rich delivery record

- When the closeout docs are rendered, embed the full `sdlc_delivery.json`
  payload (as a `"delivery"` object, schema tag
  `autotrain_sdlc_delivery/v1` preserved) inside the per-cycle results JSON
  the driver already writes to `docs/design/` — so future mining gets
  `candidate_id`/`arm_seed`/`policy_sha256` without reasons-string
  recovery. Do not write a second file; extend the existing JSON payload.
- Extend `extract_observations` to read this embedded shape (and the four
  existing standalone `autotrain_sdlc_delivery/v1` docs): slug via the
  existing `slug_from_candidate_token` on `candidate_id`, seed from
  `arm_seed`, delta from its metric pairs, dedupe keys unchanged. Rebuild
  determinism (`--check`) must hold.
- Regenerate the committed artifact
  (`/home/user/slm-training/.venv/bin/python -m scripts.build_evidence_ledger --write`
  run from YOUR worktree with its own `--design-dir <worktree>/docs/design`
  and `--out <worktree>/src/.../evidence_ledger.v1.json`) and commit it.

### L3 — operator status

- Extend `scripts/build_evidence_ledger.py --status` to print, after the
  per-arm table: the current eval key, the posterior-UCB top-10 ranking over
  all ledger arms (reuse `rank_arms_by_evidence` + `current_eval_key`), and
  the `power_feasibility_report` for (n = 3, alpha = 1/20) alongside the
  required pooled n — labeled as the current screening geometry. Pure reads;
  no new flags required beyond enriching `--status`.

### L acceptance

New tests in `tests/test_autoresearch/test_evidence_ledger.py` for L1
(weight table) and L2 (embedded-delivery extraction, dedupe, determinism);
`--check` passes against your regenerated artifact; full
`tests/test_autoresearch/` green; ruff clean.

---

## Integrator contract (after all three branches report)

1. Merge `claude/df-model-levers`, `claude/df-governance`,
   `claude/df-ledger` into the delivery branch; resolve conflicts (expected
   only in `run_autotrain_continuous.py` — units touch disjoint regions —
   and none elsewhere by construction).
2. One consolidated `versions.json` bump: `harness.autoresearch.experiment_campaign`
   (driver/schemas/policy/ledger changes) plus the components watching
   `twotower.py` (`model.twotower`), `config.py`/`factory.py`
   (`harness.model_build.train` — verify with the checker's error output),
   and `levers.py` (`config.levers`), each with a history note; register any
   brand-new files as watched paths.
3. Regenerate the evidence ledger once post-merge; run
   `--check`.
4. Update `docs/design/darkfactory-hillclimb-optimization.md` (mark the
   delivered items, remove stage language) and
   `docs/design/autotrain-climb-policy.md` (v8 `terminal` block).
5. Gates: full `tests/test_autoresearch tests/test_scripts/test_run_autotrain_continuous.py
   tests/test_models tests/test_versioning`, `ruff` on all touched files,
   `python -m scripts.verify_version_stamps --check`,
   `python -m scripts.repo_policy`,
   `python -m scripts.refresh_test_cases --check --changed`.
6. Commit, push, open the PR (ready for review), subscribe to its activity.
