# S9 / H7: preregistered adaptive-depth campaign for the shared recursive denoiser

**Status: `blocked_on_fixture_headroom`** — the `ExperimentCampaignV1` manifest is
locked (digest below) but execution is refused until the fixture-headroom
precondition is satisfied and the control re-locked. Filing is not evidence.
No training, evaluation, or benchmark was run for this record.

- Manifest (envelope + locked `ExperimentCampaignV1`):
  `src/slm_training/resources/experiments/recursive_adaptive_depth_h7/campaign.v1.json`
- Arm configurations (digest source for every `config_sha256`):
  `src/slm_training/resources/experiments/recursive_adaptive_depth_h7/arm_configs.v1.json`
- Locked manifest digest (`campaign_manifest_sha256`):
  `8a5710dc2186aead126938bdd64edacef158f1ace61b7598ebb953d2c577492c`
- Claim class: `fixture` (fixture-only diagnostic; never a ship, checkpoint,
  or production-default claim).
- Source commit at lock: `afa5f81b4e545f4c9ad0f9a1b8ff1d87f29d0e0e` (clean).

## Hypothesis under test (external hypothesis H7)

`SharedRecursiveDenoiserTower` (`src/slm_training/models/recursive_denoiser.py`)
applies one shared map R times with R fixed at construction; the per-depth
telemetry `y_update_norm`, `z_update_norm`, `y_update_state_ratio` on
`RecursiveDepthDiagnosticsV1` is emitted but never read back, and no halting or
adaptive depth exists. H7 proposes:

1. hide the recursion index from the tower;
2. decay per-depth supervision to zero at the final depth (an EqM-style
   `c(R) -> 0` analogue); and
3. halt decoding when `y_update_state_ratio < tau` instead of running a fixed R.

### Locked interpretation of "index-hidden"

A read of `recursive_denoiser.py` finds no recursion-index embedding or
conditioning on the forward path: the shared map receives only the current
`(y, z)` state. The index enters the tower today only through (a) the
per-depth supervision weight schedule (`recursive_depth_supervision_weights`,
indexed by depth) and (b) the fixed-R stopping rule. The manifest therefore
locks "index-hidden" as: **no per-depth conditioning may be added to the
tower** (`recursion_index_conditioning: "none"` in every arm config), the
supervision schedule is the only index-dependent training signal and it decays
to zero at `r = R`, and the stopping rule becomes state-driven (halting arm) or
stays fixed (control). Any arm that adds an index input violates its locked
`config_sha256` and is a `CampaignDeviationV1`, never confirmatory.

### Decayed depth supervision

Both candidates use the canonical `recursive_depth_aux_mode="intermediate_only"`
(`RECURSIVE_DEPTH_AUX_MODES`, `src/slm_training/models/twotower.py`), whose
eligible depths are `1..R-1` and whose final depth carries only the primary
loss, i.e. the auxiliary weight at `r = R` is exactly zero by construction.
The locked schedule is `c(r) = (R - r) / R` for `r < R`, so with `R = 4` the
weights are `(0.75, 0.5, 0.25)`; `ValidatedDepthSupervision.normalized()`
divides by their sum, which preserves the linear decay.

## Arms (size-matched; halting adds no parameters)

All three arms share the SLM-421 `_build_recurrence_health_model` recipe
(`scripts/run_slm138_recursive_denoiser_fixture.py`: `d_model=32`, `n_heads=2`,
`context_layers=1`, `denoiser_layers=2`, `denoiser_arch="shared_recursive"`,
`recursive_transition_layers=2`, `R = 4`, AdamW lr 1e-3 for 4 steps, CPU,
fixture records digest `480ed389…47aa`). `levers.require_size_matched_arms`
must pass for every candidate against the control; `EG_params` passes
trivially because neither the schedule nor the halting rule owns a parameter.

| arm_id | role | depth supervision | halting | config_sha256 |
| --- | --- | --- | --- | --- |
| `fixed_r_control` | control, `mechanism_off_arm_ids` | `off` (SLM-421 `as_is`) | none, fixed R=4 | `4fe902f3…55ef` |
| `index_hidden_decayed_supervision` | candidate | `intermediate_only`, `(0.75, 0.5, 0.25)` | none, fixed R=4 | `ceaadd5e…ec53` |
| `index_hidden_halting_tau` | candidate | `intermediate_only`, `(0.75, 0.5, 0.25)` | decode-only: stop after step r when `y_update_state_ratio_r < tau`, `tau = 0.30`, `r_max = 4` | `f44b110e…178b` |

Two candidates require a locked `selection_rule`; it is
`best_by_primary_then_smallest`.

### Why `tau = 0.30`

Preregistered from SLM-421's `as_is` telemetry, not from any H7 outcome: the
batch `y_update_state_ratio` at depth 1 was ~0.71–0.79, at depth 2 ~0.45, at
depth 3 ~0.32, and at depth 4 ~0.25 across seeds. `tau = 0.30` therefore sits
between the depth-1 band (so halting at depth 1 on a majority of inputs would
be a genuine failure, not a threshold artefact) and the depth-4 band (so the
rule can fire before `r_max`). The two threshold-direction kill criteria below
make either failure mode executable.

## Endpoints

- **Primary** `adjacent_depth_ce_nonregression_pass_rate`: per-seed pass iff
  `CE(r+1) <= CE(r) + 1e-3` at every `r < R_eff` for every fixture example (the
  SLM-421 recurrence-health condition, read from the single post-training
  diagnostics forward as in SLM-421). Decision by the SLM-421 Wilson power
  rule with `min_pass_rate = 0.5` locked before any seed is read.
- `per_depth_ce_paired_delta_max`: per-depth CE non-regression **vs the
  control at every depth** `r = 1..4`.
- `final_depth_ce_paired_delta_vs_control`: paired final-depth CE (the halted
  depth for the halting arm) against the control at `R = 4`.
- `steps_to_halt_mean` plus the full per-input steps-to-halt distribution as a
  required artifact.
- `halt_trigger_fraction`, `halt_at_depth_1_fraction`.
- `wall_clock_seconds_per_seed` (all three arms per seed; must fit
  `MAX_RUN_MINUTES = 3` over 20 seeds, hence the 9 s/seed gate).

Holm family `h7_confirmatory` (alpha 0.05) covers the primary and the two
paired-CE endpoints; the halting and wall-clock endpoints are descriptive and
gated but not multiplicity-corrected.

## Seeds and power

Twenty fresh seeds `22..41`, disjoint from `0..21` (SLM-282 used 0/1; SLM-421
used 2..21; SLM-434 used 2..6). SLM-421 ran 2 arms × 20 seeds × depths
{1, 2, 4} inside the 3-minute cap on the shared CPU host; this campaign trains
3 arms × 20 seeds at a single `R = 4`, i.e. 60 trainings against SLM-421's
120, so the cap is expected to hold. SLM-434's declared deviation (20 seeds
infeasible on the SLM-317 decode path) does not transfer: that cost was
per-seed AR decoding, absent here. A run that exceeds the cap is aborted and
is never evidence; no seed-count deviation is pre-authorised.

## Executable kill criteria (`executable_kill_criteria`)

1. `any_depth_ce_regression` — as SLM-282 (seed 1, `R = 4`, example `b`, CE
   17.487688 → 17.855324): Wilson 95 % upper bound of the per-seed pass rate
   below 0.5 kills the arm.
2. `per_depth_ce_regression_vs_control` — any depth with mean paired CE above
   the control by more than 1e-3 kills that candidate.
3. `halting_never_triggers` — `halt_trigger_fraction == 0` over 20 × 2 inputs.
4. `halt_at_depth_1_majority` — `halt_at_depth_1_fraction > 0.5`.
5. `size_mismatch` — `require_size_matched_arms` fails for any candidate.
6. `fixture_headroom_missing` — refuse to start (see below).

Each has a matching rollback gate in the manifest so the kill is a gate
evaluation, not a narrative.

## Controls

- Positive `size_matched_fixed_r` (the control arm itself).
- Negative (destructive) `tau_inf_halt_at_depth_1`: re-decode the halting arm
  with `tau = +inf`; it must halt at `r = 1` on every input, reproduce the
  control's depth-1 CE bit-for-bit, and trip the `halt_at_depth_1_majority`
  kill. If it does not, the halting rule or the kill criterion is not live and
  the campaign is void.
- Quality `slm421_control_reproduction`: the control on seeds 22..41 must pass
  the SLM-421 condition at a rate whose Wilson interval overlaps SLM-421's
  `[0.699, 0.972]`.
- Quality `fixture_headroom_precondition` — the blocking precondition.

## Fixture-headroom precondition (why the campaign is blocked)

The manifest requires, before any seed runs, that the eval fixture contain at
least one eval example on which `ar_only` is invalid. Search performed on
2026-09-02:

- `scripts/run_slm138_recursive_denoiser_fixture.py::_fixture_records` — two
  synthetic records `a` ("Hero layout") and `b` ("CTA layout"), both
  `split="train"`, used as the recurrence-health evaluation set. The fixture
  has no `ar_only` decode arm at all, so no example can be `ar_only`-invalid
  on it.
- `docs/design/iter-slm317-repair-hybrid-powered-rerun-20260727.md` (SLM-434)
  — on the SLM-155 eval decision corpus (`make_fixture_decisions(n=n_eval,
  seed=1)`), `ar_only` is hard-valid on 40/40 paired outcomes: "the frozen
  fixture leaves no headroom".
- `docs/design/iter-slm282-recurrence-health-powered-rerun-20260725.md`
  (SLM-421) — examples `a`/`b` only; no validity headroom is recorded.
- No `docs/design/iter-slm434*` file exists; SLM-434's record is the SLM-317
  powered-rerun document above.

**Result: no such example exists.** The campaign is therefore marked
`blocked_on_fixture_headroom` in the envelope, in `stopping_rules`, in the
`fixture_headroom_precondition` control, and in the
`fixture_headroom_missing` kill criterion. Unblocking requires a successor
fixture with recorded `ar_only`-invalid headroom (SLM-434 assigned that to a
future decision issue), a new `locked_eval_manifest_sha256`, and a re-lock of
this manifest; the current digest may not be reused for that run.

`locked_eval_manifest_sha256` in the manifest is the SLM-138 fixture records
digest (`_records_hash(_fixture_records())`), not a held-out evaluation
manifest; it binds the two frozen records, nothing more.

## Registration and validation

The RESEARCH-02 preregistry
(`src/slm_training/resources/research_experiment_preregistry.json`, verified by
`scripts/verify_research_experiment_preregistry.py`) accepts only the
`RESEARCH-03..20` pilot keys, so this campaign is not an entry there; it is
registered as a resource manifest alongside the other experiment resources
under `src/slm_training/resources/experiments/`. The preregistry step of
`verify_merge_ready --fast` is unaffected by this change and stays green.

Validate (loads the manifest through `ExperimentCampaignV1`, recomputes the
digest, and constructs a `CampaignLockV1`):

```
PYTHONPATH=$PWD/src python -c "import json; from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1, CampaignLockV1, campaign_manifest_sha256; d=json.load(open('src/slm_training/resources/experiments/recursive_adaptive_depth_h7/campaign.v1.json')); m=ExperimentCampaignV1.model_validate(d['manifest']); assert campaign_manifest_sha256(m)==d['manifest_sha256']; CampaignLockV1(manifest_sha256=d['manifest_sha256'], manifest=m); print('ok', d['manifest_sha256'])"
```

Nothing in this record is an outcome. When a run happens it must follow
`documenting-experiment-results` and land its own `iter-*.{md,json}` pair
stamped against `model.recursive_denoiser`.
