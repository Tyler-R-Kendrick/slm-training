# Autonomous experiment contracts — local implementation evidence

## Finite coordinator comparison (September 7 local time)

This section supersedes earlier worker handoff status, not historical results.
Canonical CLI registration is `python -m scripts.slm experiments learning-comparison`;
the actual finite commands invoked its registered module directly:

```text
python -m slm_training.harnesses.experiments.autonomous_learning.cli prepare --run-id autonomy-learning-current-20260907 --enable-fixture-experiment
python -m slm_training.harnesses.experiments.autonomous_learning.cli fidelity --run-id autonomy-learning-current-20260907 --enable-fixture-experiment --treatment <fast|steady|slow> --updates <1|4>
python -m slm_training.harnesses.experiments.autonomous_learning.cli corrective --run-id autonomy-learning-current-20260907 --enable-fixture-experiment
```

Execution order was prepare; fast/steady/slow at one update; fast at four
(failed before continuation); repaired fast/steady/slow at four; corrective.
Every invocation used `timeout -s INT -k 10 170`, `PYTHONPATH=src`, one Torch
thread, the already installed bridge/AgentV runners and local CPU scratch.
The registered procedure locked seed 7301, 17 total optimizer updates and public
AST-copy fixtures before outcomes. Model and data identities are in the
[model card](../MODEL_CARD.md#finite-learning-and-fidelity-fixtures--2026-09-07).

The failed fast continuation reproduced the original initializer-path mismatch
in the actual trainer. Train v44 treats that path as initialization provenance,
not an instruction to reinitialize a restored optimizer. A controller-owned
manifest permits exactly the reviewed old/new source digests; all other source,
recipe, data, tokenizer, optimizer, RNG and runtime checks remain. Its SHA-256 is
`1a6387f1de50d8fe6112b023b4e2c331468774c6d0b22d3ad8e7a1e72cb41c8d`.
The failed input remains immutable. Event
`3b7ee3439717da756c5059e8cfbc49d315b1d6ad79c166ee4a04613f17c58ba3`
links its release successor without claiming verification; actual training then
validated and completed the continuation. The failed 0.532438-second attempt
remains charged. No new seed, treatment, checkpoint initialization or budget reset.

| Fidelity treatment / LR | CE after 1 update | CE after 4 total updates |
| --- | ---: | ---: |
| fast / .03 | 16.743950 | 12.311351 |
| steady / .003 | 29.390649 | 20.532096 |
| slow / .0003 | 31.020020 | 30.507930 |

All six checkpoints produced two complete decoded rows plus actual AgentV output.
The three-configuration rank agreement was 1.0, selection regret 0, consumed
fidelity updates 12. This tiny fixed fixture did **not** exhibit a late-blooming
ranking reversal and does not establish safe low-fidelity pruning. Rotation
remains the fixture reference; production search/adoption policy is unchanged.
CE is conditional masked-token loss, not exact sequence likelihood. Node 26.5.0
ran the lower checkpoints' bridge/publication path; installed supported Node
22.23.1 ran the higher and corrective paths. Python/Torch, loss estimator and
model/grammar identities matched, but cross-Node decoded/latency equivalence was
not established. No semantic or latency fidelity conclusion is drawn.

Original/corrective datasets contain 2/4 admitted rows from the same two train-only
roots; the two scored roots differ. Quality reports, empty rejection files and
correction feedback were read. Two verified oracle corrections add no independent
root families. Correction arms match two updates and four examples, not prompt
tokens (144 control / 288 candidate). Both are 64,962 parameters. Two paired eval
roots give mean control-minus-candidate CE **−0.00346847**, paired SD .00118626,
Wilcoxon p=1, **inconclusive** (two non-tied pairs below the existing floor).
Choice-focused oracle-assisted CE is 31.562942 / 31.565129. No confirmed gain;
this is not a useful-effect exclusion. Both decoded arms have binder F1 zero
and two certified fallback outputs; syntactic validity is not semantic learning.

All planned fixture updates completed: 17 incremental updates, 34 examples,
1,368 prompt and 272 target tokens. Lower checkpoint consumption is included in
the final four-update cursors, never added again. This is a real multi-invocation
training continuation, not a demonstration that each training chunk needs three
minutes. Invocation caps were preserved; complete logical end-to-end wall/resource
accounting is still an integration obligation, not inferred from train timers.
No confirmation or production suite was run, no champion moved, no service started.

Current source tests: `pytest -o addopts= tests/test_harnesses/experiments/test_learning_fixture.py tests/test_harnesses/model_build/test_trial_continuation.py tests/test_scripts/test_merge_verification.py --junitxml=outputs/runs/autonomy-validation/release-successor-resume.xml -q`
passed **45 tests in 50.93 seconds**. This includes real scratch/warm-start
six-update versus 3+3 exact-state comparisons and separate receipt/cache tests.
The subsequent bounded static verifier passed all 18 checks; static success is
not full merge authorization. Raw result hashes and limitations are recorded in
the companion JSON; no fake agent result supports these model observations.

## Current restored-candidate verification

### Integration patch handoff (current)

Final proposed-source regression: **109 passed, zero skipped, exit 0, 16.76s**;
`outputs/runs/exp-mixin-overlay-final.xml`. Effective options clear `addopts`.
Exact commands, current/proposed source digests and all patch hashes are in the
companion JSON's `integration_update`. Owned-path Ruff, complexity, case
extraction and diff checks pass. Whole-tree quality remains the parent's open
integration obligation; no baseline was raised.

Ready `apply_patch` inputs in the candidate workspace:

- `outputs/runs/autonomy-exp-role-integration.patch`: default five-candidate
  search; exactly two confirm/promotion arms, no +1000/+2000 monitor recipes.
- `outputs/runs/autonomy-exp-launch-integration.patch`: canonical training CLI
  config resolution without training; validated intervention/resource fields;
  mandatory `compiled_treatment.lock_driver_designs` and attempt events before
  actual launch. Ordinary warm-start guards remain for mechanism arms.
- `outputs/runs/autonomy-exp-role-test.patch`: update the existing confirmation
  regression's obsolete padding expectation (the search floor is unchanged).
- `outputs/runs/autonomy-checkpoint-modelcard.patch`: exact v43 local paths,
  bundle digests, recipe, unevaluated status and README summary.
- `outputs/runs/autonomy-exp-legacy-serialization.patch`: preserve omitted legacy
  identity/role fields on round-trip serialization; no frozen payload rewrite.
- `outputs/runs/autonomy-exp-schema-ownership.patch`: after the preceding patches,
  move identity/role fields and migration hooks into the existing identity owner.
  Retains strict/frozen model settings and matrix membership validation; reduces
  patched schema from 1,672 to 1,648 lines. Parent retains the remaining shared
  baseline-budget reconciliation.

Shared originals were not edited by this worker. All three implementation/card
patches applied successfully to isolated copies under
`outputs/runs/exp-patch-validation.lTjcRi`; imported proposed-source overlay tests
passed 40 checks in 19.96 seconds, including real model construction/parameter
counts and canonical campaign events. This is **proposed-source integration**,
not a real supervised training comparison or activation of the parent's tree.
The parent subsequently reported applying the five initial patches; the sixth
ownership patch is supplied separately and was tested in the same isolated copy.

`preflight/compiled_treatment.py` uses the real compiler, CLI resolver, model
factory, tokenizers, loader and typed matching/preflight owners. It locks ordered
NLL/decode selection separately and binds source/runtime, data, ancestor,
endpoint, resource contract and hypothesis/treatment/replicate/attempt identities
to canonical artifacts/events. Local CPU scratch preparation is supported;
unavailable HF/GPU capability and unpublished training snapshots refuse explicitly.
Uniform example-budget comparisons account for final partial batches; unsupported
nonuniform planners refuse rather than inventing equal work. Attempts do not add
independent samples. No preflight or process exit authorizes promotion.

The first patch-test run found invalid test-only knob names and an unversioned
eval identifier; tests now use the actual strict schema and DataStore root.
Its failure receipt remains `outputs/runs/exp-proposed-patch-tests.xml`.

Original evidence below is historical; its vanished `/tmp` artifacts are not
current validation. The authoritative candidate is
`/home/codex/repos/slm-training/outputs/workspaces/autonomy-candidate`.
New current receipts are recorded in the companion JSON. Hypothesis now executes
using the existing invocation-local `outputs/runs/autonomy-validation/dependencies`
installation; this worker installed no dependency and no longer reports its old
dependency skip as current coverage.

`arm_trainable_params` now consumes the actual trainer's
`track.trainable_params`, rejects malformed authoritative counts rather than
falling through to a convenient alias, and preserves positive legacy integer
counts. `arm_completed_n` rejects booleans, fractional/negative/nonfinite/string
counts. Regressions call those real consumers. The treatment identity golden is
`ef2296d0e653659ac06a9248c1f07b4f5f9389f7b7d045df18251658d175fa69`;
the older JSON golden remains historical, not a compatible new identity.

The ownership test tables were extracted with the existing case-file generator.
No scientific threshold or source authority changed. Parent-owned wiring still
must require the resolved treatment preflight verdict, persist treatment/replicate/
attempt identities, and apply the role-aware matrix hook below. This report does
not turn an owner-local passing test into a completed supervised experiment.

Final combined domain invocation: **93 passed, zero skipped, exit 0, 26.99s**,
`-o addopts=`, receipt `outputs/runs/art-exp-champion-final.xml`. Owned-path Ruff
and targeted case extraction pass. `refresh_test_cases --check` selected no files,
so it is not claimed as a substantive new-resource review. Source digests and
remaining integration obligations are in the JSON's `latest_domain_handoff`.
The current confirmation producer still contains `+1000/+1001/+1002`; the
parent-owned changes below remain executable obligations, not completed wiring.

## Historical implementation record

Base: `e0eca9f9910244ecc20eb480d363f1852417b599`; audit baseline:
`2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`. This is a shared dirty authoring
tree, not a committed/released source identity. No Git, Actions, HF, cloud,
service activation or scientific gate changes were performed by this owner.
Machine evidence: [autonomy-experiments-20260907.json](autonomy-experiments-20260907.json).
ART handoff precedes this work: [artifacts evidence](autonomy-artifacts-20260907.md).

## Implemented ownership and migration

- EXP-IDENTITY: `autoresearch/experiment_identity.py` owns strict JSON treatment
  identities, randomized planned-unit identities and attempt ordinals. Resolved
  training length, batches, data, architecture, tokenizer layout, ancestor role,
  endpoint and resource contract remain material. Incidental retry/run IDs and
  nuisance seed realizations do not create new treatments. Seed realizations
  belong to the replicate. An explicit randomness-distribution intervention is
  material. The identity is not an independence or authenticity certificate.
- Existing `scripts/autotrain_levers.py::knobs_fingerprint` remains the legacy
  step-excluding reader; its history is not rewritten. New work uses
  `resolved_treatment_identity`. `legacy_identity_reference` explicitly labels
  unresolved old identities, which cannot authorize compatible pooling.
- EXP-INVARIANTS: `assert_declared_arm_match` delegates to the canonical new
  intervention matching helper. Exact declared changes only; common planned
  resource unit and measured parameter counts required. Data and duration
  interventions are explicit exceptions, never blanket removal of ordinary
  `assert_warm_start_launch`. Changed checkpoint/seed/tokenizer remains refused.
  A capacity experiment requires charging and still needs authoritative
  size-normalized promotion evidence; this helper never promotes anything.
- EXP-CAUSAL/PREFLIGHT: explicit `lever_effects.bank_effects()` metadata overlays
  the existing bank categories (including training compiler losses previously
  categorized by name). Existing canonical constraint/capability/companion
  validators run unchanged. Unregistered knobs, unchanged knobs, wrong endpoint
  families, disabled owning objectives and dropped compiler flags are refused.
  `preflight/treatment_design.py::CHECK` is automatically discovered through the
  existing preflight runner; no new scheduler, store or verdict authority.

## Production integration contract (parent-owned edits)

The parent exclusively owns the shared schema, policy, entrypoints and version
registry. These are required composition edits, not assertions that this domain
worker changed those files:

1. `HypothesisMatrix`: add
   `matrix_role: Literal["search", "screen", "confirm", "promotion"] = "search"`;
   change the structural tuple floor to `Field(min_length=2)`; call
   `validate_matrix_role(self.matrix_role, len(self.hypotheses))` from the existing
   validator. Default `search` preserves legacy five-candidate validation. Other
   roles require exactly two arms. Keep recommendation/member/uniqueness checks.
   Previously frozen five-row confirmation artifacts stay legacy search-shaped
   artifacts; do not mutate them into two-row manifests.
2. `_matrix` confirmation producer: delete the three monitor-only rows with
   `confirm_steps + 1000`, `+1001`, `+1002`; set `matrix_role="confirm"`. The
   current real imported producer emitted `[80,80,1080,1081,1082]`, and
   `HypothesisMatrix.model_validate` accepted it. Update the existing
   `test_matrix_confirm_path_same_levers_new_seed` assertion from `len>=5` to
   exactly two. Parent already removed `cycle % 3`; do not recreate it elsewhere.
3. At actual compiled launch, supply `candidate["treatment_design"]`:

   ```python
   {
       "control": resolved_control_config,
       "candidate": resolved_candidate_config,
       "intervention": {
           "kind": "mechanism",  # or declared data/duration/capacity/randomness
           "varied_fields": ["lr"],
           "resource_basis": "updates",
       },
       "resource_totals": [20, 20],
       "trainable_params": [64546, 64546],
       "endpoint": "denoising_loss",  # decoded_quality or latency
       "compiled_commands": actual_compile_commands_output,
       "bindings": {
           "architecture": architecture_identity,
           "tokenizer_layout": tokenizer_layout_digest,
           "training_snapshot": training_manifest_digest,
           "preprocessing": preprocessing_digest,
           "starting_checkpoint": checkpoint_bundle_digest,
           "starting_checkpoint_role": "warm_start",
           "endpoint": versioned_endpoint_identity,
           "resource_contract": locked_scientific_resource_contract,
       },
   }
   ```

   Values above are structure examples; no fabricated dataset/parameter identity
   is permitted. Materialize full effective configurations before hashing; sparse
   proposal deltas are insufficient. Resource quantities are locked controller
   inputs, not claimed post-run consumption; reconcile actual ART accounting.
   `updates` quantities are additionally checked against recipe steps.
   Missing legacy design yields `warn/legacy_unresolved_design` with
   `release_authorized=False`. The parent must require the mandatory design check
   to be present and `pass` before new training work launches, rather than using
   `not has_block` as merge/science authority. Imports/check crashes in the old
   optional discovery runner likewise do not discharge this mandatory check.
4. Persist `treatment_id` from the passing verdict, separate `replicate_id` from
   locked design/root/seed/ancestor, and `attempt_id` from retry ordinal in
   existing campaign artifacts/events. Reuse old indexes only as explicitly
   legacy diagnostics, not novelty or independent evidence. A seed forked from a
   learned ancestor supports a conditional-on-ancestor claim only.
5. On a **declared** data/duration launch call `assert_declared_arm_match` with
   resolved control/candidate configs; otherwise retain ordinary warm-start
   validation. Data readiness/leakage and existing promotion gates still run.
6. Component/ownership registration: include new autoresearch identity/matching,
   lever-effect and preflight files in the existing experiment/campaign owner;
   bump affected component versions rather than relabeling old evidence.
   `levers.py::lever_catalog` may expose `bank_effects` metadata through a lazy
   call, keeping the existing registry authoritative. Avoid an eager import
   back into ModelBuildConfig. No registry was edited by this worker.

## What actually ran

All commands used the supplied venv, `PYTHONPATH=/tmp/slm-autonomy-20260907/src`,
local `XDG_CACHE_HOME=outputs/.exp-cache`, `PYTHONDONTWRITEBYTECODE=1`, and
`timeout -s INT -k 10 170`. CPU neural work used one Torch/OMP/MKL thread.
No evaluation metrics/AgentV scores were claimed by these training wiring tests.

```sh
timeout -s INT -k 10 170 env PYTHONPATH=/tmp/slm-autonomy-20260907/src \
  PYTHONDONTWRITEBYTECODE=1 XDG_CACHE_HOME=/tmp/slm-autonomy-20260907/outputs/.exp-cache \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/codex/repos/slm-training/.venv/bin/python \
  -m pytest -q -o addopts= tests/test_autoresearch/test_experiment_identity.py \
  tests/test_autoresearch/test_experiment_identity_property.py \
  tests/test_autoresearch/test_experiment_lever_wiring.py \
  --basetemp=outputs/experiment-contract-tests-final --tb=short --show-capture=no
```

Before the final golden assertion: **34 passed, one explicit dependency skip,
3.54s, exit 0**. Marker defaults were cleared, so the `training` test executed.
The final rerun is recorded in the JSON manifest. Independent source-related
compatibility selection:
`test_preflight_gates.py test_run_autotrain_continuous.py -k 'fingerprint or bank or lever or matrix_confirm_path or steps_confirm'`
gave **32 passed, 357 deliberately unselected, 2.51s, exit 0**. This is focused
coverage, not a merge authorization receipt.

Actual compiler-to-CLI tests observed `ModelBuildConfig` for lr, steps, batch,
LTR-tail loss and ambiguity-only loss. They used a valid isolated train fixture;
only the final train call was replaced to inspect config. A separate real CPU
TwoTower test trained both LR arms from matched seed/config for two updates.
LRs 0.0003 vs 0.003 consumed three examples per arm; 44 of 48 state tensors
differed, maximum absolute tensor difference 0.005906892940402031. This
demonstrates executed parameter wiring, **not** scientific quality improvement.
Transient fixture checkpoints live under `outputs/experiment-contract-tests-*`.
Model-card/README integration note: CPU scratch wiring fixture, 64,546 trainable
parameters, two updates, no decoded evaluation, no promotion or ship claim.

## Failures retained and limitations

- Initial collection: missing Hypothesis, exit 2. Its declared dependency is not
  installed in the supplied venv. Generated property test remains an explicit
  skip; fixed-seed metamorphic tests run. No dependency installed or upgraded.
- First wiring run: five parser tests accidentally resolved the default corpus
  with open synthesis feedback, and one fixture passed duplicate LR kwargs.
  Fixed test inputs to the isolated fixture and used dataclass replacement;
  no corpus feedback or production gate was waived. Then all executed tests
  passed.
- Actual real `_matrix` padding defect is reproduced in current code; removal
  belongs to parent integration. Identity tests exercise the real new owners,
  not transcribed copies of historical functions. New identity semantics did
  not exist in the baseline; no claim that these tests ran against an old API.
- No all-lever neural gradient proof, statistical improvement, independent
  confirmation, promotion or full service-autonomy claim. The metadata covers
  explicit supported keys; unknown/new mechanisms require owner wiring evidence.
- Data/measurement authority, complete driver wiring and index migration remain
  parent composition obligations. Legacy optional preflight warnings alone do
  not enforce the new default. Existing frozen artifacts are unchanged.
