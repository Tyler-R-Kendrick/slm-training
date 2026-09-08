# Autonomous training artifacts: implementation and measured evidence

## Current restored-candidate evidence

### Coordinator integration correction

Baseline seeding now resolves the checkpoint named by the actual in-run
`train_summary.json`, rather than preferring `checkpoints/last.pt`. The latter
can be a loose compatibility artifact without the bundle's exposure history.
The driver regression removes that loose alias and still seeds from the
validated bundle; foreign/missing recorded paths are not silently substituted.
The combined driver/data/activation/ONNX/lineage/operations check passed 433 tests
with one skip in 50.97 seconds. Storage fixture weights and exposure counters
are explicitly synthetic; the retained real CPU continuation evidence below
remains separately classified. No new champion was published by these tests.

Current shared owners are `harness_core/checkpoint_bundle.py`,
`harness_core/checkpoint_exposure.py`, and `harness_core/checkpoint_publication.py`.
Older paths in historical receipts below identify the source tested at that time.

### Exposure consumer completion (current)

Final proposed-source/domain suite: **109 passed, zero skipped, exit 0, 16.76s**
(`outputs/runs/exp-mixin-overlay-final.xml`). It also covers confirmation of an
existing baseline bundle: claim status may change after confirmation, but updates
and exposure do not increase. The real retained v43 chunk2 bundle was separately
read by the new consumer and yields 18 examples/546 target tokens/six epochs.
Source identities and exact commands are in `integration_update` in the JSON.

Champion writers now bind validated per-snapshot trainer exposure through
`model_build/checkpoint_exposure.py`, using `climb_champion/v3`. Existing v1/v2
records remain legacy estimates; a missing schema stays v1, never acquires v3
measurement semantics. New bundles with unknown ancestor consumption remain
explicitly unmeasured (`champion_exposure_unavailable`). Each snapshot retains
its own denominator. Resumed cumulative counters are not added to the previous
champion, and replaying the same committed training bundle does not add updates.
Baseline seeding and confirm-only advancement guards remain unchanged.

Real bundle-to-champion regressions demonstrate 100 updates x two examples on
ten records = 20 exposure epochs; a later 100-example/100-record snapshot adds
one epoch without erasing the earlier 20. Missing/malformed counters refuse
strong measurement. Tests live in `test_checkpoint_exposure.py`, not the
oversized generic hillclimb test. Combined direct consumers/publication suite:
34 passed in 10.41s (`outputs/runs/art-exposure-consumers-v2.xml`). The preceding
30-pass/four-failure run exposed the parent's transient CampaignLockV1 import
regression, now repaired; it is not silently discarded.

The parent reports fixing the child startup race with a bounded registration
gate. This worker's ordinary child tests pass, but its earlier report did not
independently test delayed registration. The earlier numeric-consumer limitation
below is superseded by this direct implementation and its focused regressions.

The implementation record below and the original JSON fields are **historical
recovered receipts**: the old `/tmp` payloads were lost. Current authoring and new
payloads are in `/home/codex/repos/slm-training/outputs/workspaces/autonomy-candidate`.
See the companion JSON's `current_verification`; no historical receipt authorizes
this candidate.

Final fresh regression: **93 passed, zero skipped, exit 0, 26.99 seconds**;
`outputs/runs/art-exp-champion-final.xml`. Effective pytest options explicitly
clear `addopts`. The real child calls `hillclimb.write_climb_champion` and reads
the published sidecar with `load_climb_champion`; it does not merely call a mock
publisher. Owned-path Ruff passes. Latest source digests, test receipt and the
unchanged operational fixture plan digest are in `latest_domain_handoff`.
This is domain coverage, not independent whole-release authorization.

The fresh separate-process comparison found a real additional defect: weights,
optimizer and Torch RNG matched, but Python/NumPy RNG did not. The trainer now
calls `full_state.seed_training_rngs` before preparation and restores saved streams
on resume. Missing RNG/scaler/scheduler fields fail exact-resume validation.
Scaler requirements follow the effective scaler, not an unsupported CPU AMP flag.
The expanded actual-trainer regression perturbs process RNG before each run.

The champion scope accepts RUNTIME's `DelegatedPublisher(store, source_digest)`.
Every publication re-enters `publisher.publication(lease)`; expired scope cannot
fall back to legacy local CAS. Actual child-process tests cover publication with
the current fence, missing delegation, and cancellation. Their dummy component
bytes prove integrity plumbing, not model quality. No second adapter was added.
The parent supplies the supervisor scope and protected controller storage.

Fresh train component **v43** evidence: CPU scratch TwoTower, 64,546 parameters,
seed 0, LR .003, width 32, one context/denoiser layer, batch 2, accumulation 2,
lexer output; three public regression-fixture records (two related Hero rows).
No independent-data or semantic-quality claim is made.

| Run (`art-continuation-…-v4-20260907`) | Total updates | Examples | Target tokens | Process wall s | Trainer wall s |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole | 6 | 18 | 546 | 9.062349 | 1.312470 |
| chunk1 | 3 | 9 | 273 | 16.759777 | .887062 |
| chunk2 | 6 | 18 | 546 | 11.018550 | 2.039804 |

All 17 checked model/optimizer/scaler/RNG/sampler/counter fields match exactly.
The logical trial is **six**, not nine, updates: 18 examples, 546 target tokens,
six example-exposure epochs, and 27.778327 process-wall seconds across two
invocations. The reference separately costs 9.062349 seconds. The fixed-seed,
six-update plan has no adaptive selection; its logical two-invocation grant is
360 seconds. Each command uses the unchanged canonical 170-second interrupt and
10-second grace. Actual weights, tokenizers, resume state, argv and exit/duration
receipts remain in `outputs/runs/art-continuation-*-v4-20260907/`.
`outputs/runs/art-continuation-v4-result.json` records identities and exact argv;
chunk2 consumes chunk1's immutable `trial_cursor.json` resume path.

Earlier candidate runs remain separate, including the RNG mismatch and v42
receipts. Initial verification failures omitted the explicit installed bridge
paths; another ownership fixture exhausted its five-second import budget under
load. Both failures remain recorded. That non-performance fixture now grants
30 seconds plus one-second cleanup; no scientific latency endpoint or canonical
cap changed. The actual training policy/objectives and all ship gates are unchanged.

All twelve CLI fixture runs remain: 48 actual incremental optimizer updates,
10.871625 summed trainer-wall seconds. Six v3/v4 commands have full process-wall
receipts totaling 52.064120 seconds; six earlier full-process durations are
unknown, not zero. These totals exclude pytest/review/admin work. Full histories
are retained rather than charging only the final successful comparison.

Remaining integration risks: child identity is recorded by an asynchronous
`on_start` observer, so a sufficiently fast child can request its guard before
registration; passing ordinary child imports does not exclude that race. The
scope is trusted-controller plumbing, not a sandbox against same-user arbitrary
code. Parent-owned champion numeric consumers still use update-count/current-
corpus division; correct per-snapshot trainer exposure is not proof that those
consumers are fixed. No global ART-EXPOSURE completion is claimed.

No evaluation, confirmation, promotion, HF write, remote execution or old-loop
restart occurred. Parent: add the exact local fixture roster/history to MODEL_CARD
and README, record reduced quality ceilings, and finish overall release checking.

## Historical implementation record

Base: `e0eca9f9910244ecc20eb480d363f1852417b599`; directive audit baseline:
`2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`. Work is in the shared authoring clone
`/tmp/slm-autonomy-20260907`, with no commits, remote writes, service activation,
or changes to the human checkout/old loop. This domain extends the existing
`model_build.full_state`, `train_loop`, and `autoresearch.hillclimb` owners.

## Interfaces for integration

`slm_training.harnesses.model_build.checkpoint_bundle` provides:

```python
digest = stage_checkpoint_bundle(root, checkpoint, metadata, full_state=state_path)
publish_activity_bundle(runtime, lease, root, digest, expected_digest)
directory, manifest = resolve_bundle(root)
checkpoint = directory / "last.pt"
```

`publish_activity_bundle` uses the canonical `ActivityRuntime.publication(lease)`
critical section across validation and pointer CAS, excluding lease revocation.
The lower-level `publish_bundle(..., fence, validate_fence)` requires a controller
validator; a worker's claimed token is insufficient. `current_bundle_digest(root)`
supplies an expected CAS value. A repeated publication of the same digest is
idempotent only under a currently valid lease. Callers must record publication
intent in the existing campaign history before publication, then reconcile the
same digest after a crash. The helper does not create a second event authority.

Canonical `train(config)` now returns the sealed `summary["checkpoint"]`, and
`<run>/checkpoints/trial_cursor.json` names the matching immutable
`resume_from` full-state artifact, bundle digest, and total optimizer updates.
Continue with the existing `ModelBuildConfig.resume_from` / `train_model
--resume-from`; `steps` is the logical total, not additional updates. The driver
must freeze that total and charge all attempts through its resource grant.
Training publishes only a trial cursor in its attempt namespace. No quality
result, champion advancement, or shipment is implied.

`climb_champion_checkpoint_path` resolves `champion/current.json` once into an
immutable bundle. `load_climb_champion` reads metadata from that same bundle;
`write_climb_champion` stages and publishes it. Existing confirmation guards in
`seed_climb_champion` / `maybe_advance_climb_champion` are unchanged. Legacy
synchronous controller callers get local CAS; activity callers must supply the
runtime publication boundary. Numeric/metric helpers are reserved to STATISTICS.

## Contract and migration

`checkpoint_bundle/v1` has hashed weights, output tokenizer, metadata, required
context tokenizer, and optional full state. Each component is copied into a private
stage, checked for source drift, fsynced, and renamed into `bundles/<digest>`.
A fsynced atomic `current.json` points to one complete bundle. New bundles cannot
inherit an absent sidecar. Readers verify the manifest and every component;
corruption is an error. Symlink/hardlinked component inputs are refused.
This is integrity checking, not proof of a model's semantic quality or authenticity.
Worker isolation and independent acceptance remain controller responsibilities.

Legacy champion `last.pt`/`last.json` remains readable with its original schema.
New publication requires current TwoTower metadata and required tokenizers.
`climb_champion_sidecar_path` resolves to the immutable manifest for new bundles;
consumers should call `load_climb_champion` for lineage, not assume a raw JSON shape.

`full_train_state` version 2 preserves optimizer/scaler, loop/model/Torch/CUDA/
Python/NumPy RNG, pending batches, consumption counters, and an explicit zero
accumulation position. The trainer has no scheduler; this is recorded as absent.
Loading uses Torch's restricted weights-only loader. Version 1 remains readable,
but exact continuation refuses its missing compatibility proof rather than inventing
one. Weight initialization remains a separate capability.

Exact resume binds the effective recipe (excluding invocation locations, periodic
checkpoint cadence and the later total stopping cursor), published data identity
plus actual record bytes, tokenizer maps, trainer/model source hashes, Python/Torch,
architecture, device, and thread count. Mismatches require an explicit new trial
or warm start. Voluntary yields happen at optimizer boundaries. A hard kill before
the next checkpoint leaves the prior safe boundary; recomputation costs must be
charged by RUNTIME. No partial-gradient checkpoint is claimed.

Summary `exposure_by_snapshot` uses actual consumed primary/replay examples and
unpadded prompt/target token counters, with the original snapshot denominator.
It already includes accumulation and partial batch sizes. This trainer is single
worker (world size one); distributed-resume parity is not claimed. Bundle metadata
retains parent exposure for bundle-based warm starts; unknown legacy ancestry stays
`null`. Resumed cumulative counters are not summed twice.

## Executed evidence

Machine record: [autonomy-artifacts-20260907.json](autonomy-artifacts-20260907.json).
Host: Linux aarch64, Python 3.12.3, Torch 2.5.1, CPU, one thread. The real fixture
is scratch TwoTower, width 32, one context/denoiser layer, 64,546 trainable parameters,
batch size 2, accumulation 2, seed 0, three training records. No evaluation or
scientific improvement is claimed for this continuation comparison.

| Invocation | Optimizer cursor | Cumulative examples | Target tokens | Trainer wall seconds |
| --- | ---: | ---: | ---: | ---: |
| uninterrupted reference | 6 | 18 | 546 | 1.175446 |
| chunk 1, separate process | 3 | 9 | 273 | 1.243704 |
| chunk 2, separate process | 6 | 18 | 546 | 1.352758 |

Chunked logical trial: six total updates, 18 examples, 546 target tokens;
two trainer invocations total 2.596461 seconds (process startup overhead is not
included in these trainer timers). Recursive comparison of every model/optimizer
tensor, scaler, Torch/model/loop RNG, pending batches, step, and all consumption
counters passed exactly. Artifacts remain under
`outputs/artifacts-logical-trial/runs/{whole,chunk1,chunk2}/`.

The real base-source reproduction imports the entire archived base `train_loop.py`
from `outputs/artifacts-baseline/` with current dependencies. A three-microbatch
accumulation and one-token stop budget consumed 54 target tokens yet reported
optimizer cursor zero. The original owner flushed an uncounted fractional update.
This is a qualified single-owner baseline reproduction, not an untouched full-base
environment. The current regression invokes the actual current trainer and requires
one counted complete update, with five actual examples from batches 2+1+2.

The base's existing bit-exact no-accumulation test already passed (1 test, 3.98s).
The expanded owner suite initially passed 41 tests in 40.82s. Added real-process
death before/after pointer replacement, runtime epoch revocation, CAS, copy/fsync
faults, missing sidecars and corruption passed (10 bundle tests, 3.49s).
The final run including explicit legacy-state refusal passed all **44 tests in
28.55 seconds**, exit 0; ruff and `git diff --check` also exited 0.
These finite fault tests do not prove perpetual reliability or physical power-loss
semantics on all filesystems.

All invocations used `timeout -s INT -k 10 170`, derived from the canonical
three-minute limit (170-second interrupt plus 10-second kill grace), with
`PYTHONPATH=/tmp/slm-autonomy-20260907/src`, local caches and `-o addopts=`.
The final bounded test command is:

```bash
timeout -s INT -k 10 170 /home/codex/repos/slm-training/.venv/bin/python -m pytest \
  -q -o addopts= --basetemp=outputs/artifacts-tests-verified --tb=short \
  tests/test_harnesses/model_build/test_checkpoint_bundle.py \
  tests/test_harnesses/model_build/test_trial_continuation.py \
  tests/test_harnesses/model_build/test_full_state_resume.py \
  tests/test_autoresearch/test_climb_champion.py
```

The existing bridge/AgentV tests initially failed because the authoring clone had
no installed JS packages. Reruns used supported `OPENUI_BRIDGE_CLI` and
`AGENTV_RUNNER` overrides pointing to installed scripts in the human checkout.
Their bridge CLI, library, lockfile and AgentV runner SHA256 values exactly matched
this clone; those locations were read, not modified. The existing AgentV test then
passed. No fake SDK result or dependency install was accepted as evaluation success.

## Integration obligations and limits

- INTEGRATOR: register the new helper paths with `harness.model_build.train`, bump
  that component and the existing hillclimb campaign component, and run combined
  quality/version/ownership verification. Train-loop size shrinks; hillclimb size
  is unchanged before other swarms' edits; new helpers are below 400 lines.
- RUNTIME/INTEGRATOR: use the fenced helper for controller publication, bind the
  intent and terminal event to the same digest, and adopt trial cursors as activity
  outputs. The local synchronous compatibility path is not an independent lease.
- STATISTICS: consume the recorded exposure units for champion epoch policy; the
  old `champion_cumulative_epochs` current-corpus arithmetic is outside this lead's
  reserved symbols. Legacy optimizer-update totals do not establish consumed examples.
- OPERATIONS: preserve all referenced bundles and cursor ancestry in retention.
  This domain performs no GC or deletion of historical bundles. Storage pressure
  cannot authorize evicting a live trial or rollback bundle.
- Model-card/README integration note: `ART continuation fixture`, scratch CPU,
  64,546 parameters, three train records, six logical updates in two processes,
  exact state parity; local no-sync; no eval/confirmation/promotion/ship claim.
- Exact continuation from an initialization that changes tokenizer vocabulary or
  retention anchors remains fail-closed unless those extra states match; only the
  demonstrated same-recipe CPU scratch continuation is certified here. CUDA and
  distributed numerical parity, network filesystems, external authenticity, and
  loss of all durable state are not established by these tests.
