# Autotrain hill-climb recovery, round 2 (2026-09-02)

Per-card findings from the swarm working the `continuous-openui-local`
hill-climb stall. One `##` section per card. Every section states its claim
class; unless a section says otherwise, the evidence is **fixture-scale
screening** — no model was trained, evaluated, or promoted, and nothing here
is a ship claim.

Companion single-card write-ups on the integration branch:
[`autotrain-recovery-2-p9-20260902.md`](autotrain-recovery-2-p9-20260902.md).

## P1 — the control arm's exit 2 is a data-governance refusal, not a crash

Claim class: **harness change / fixture-demo**. Stamped
`harness.autoresearch.experiment_campaign v281`. No checkpoint was produced;
no gate was weakened or relaxed.

### Command and wall time

One supervised screening cycle per run, in a scratch root, under
`MAX_RUN_MINUTES = 3` per stage with a 600 s outer `timeout` as a safety net
only. Runner script:
`$SCRATCH/p1/run_merged.sh` / `run_fixed.sh`, each containing

```bash
PYTHONPATH="$W/src" timeout 600 /home/user/slm-training/.venv/bin/python \
  -m scripts.run_autotrain_continuous \
  --root "$S/autoresearch-<tag>" --loop-id p1-<tag> --max-cycles 1 --supervised
```

| Run | Tree | Campaign | Wall | Driver exit |
| --- | --- | --- | --- | --- |
| `p1-baseline` | `3238bef` (pre-merge) | `continuous-loop-20260902-p1-baseline-294c421e-c1` | 2 m 16 s | 0 |
| `p1-merged` | `a34b697` (merged, before fix) | `continuous-loop-20260902-p1-merged-f6bb706d-c1` | 2 m 10 s | 0 |
| `p1-fixed` | `92bdf49` (merged, with fix) | `continuous-loop-20260902-p1-fixed-cc16da6c-c1` | 1 m 15 s | 0 |

No run timed out, was interrupted, or was killed. Closeouts (iron law):
`continuous-loop-20260902-p1-{baseline,merged,fixed}-*-results.{md,json}` in
this directory.

### Arm exits

| Run | `control` | `semantic-contrast-compiler-margin` | Scoreboard |
| --- | --- | --- | --- |
| `p1-baseline` | 2 | 2 | not written |
| `p1-merged` | 2 | 2 | not written |
| `p1-fixed` | 2 | 2 | not written |

All three runs reproduce c536–c543 exactly: `arm_exits {control: 2}`,
`harness_failure:<arm>:experiment_failed`,
`measurement_incomplete:<arm>:missing_scoreboard`,
`primary_metric_unavailable`.

**The exit code is unchanged by this card's fix, and that is the honest
outcome** — see the root cause. The exit-2 refusal is correct behaviour of a
data-governance gate on a snapshot that really is not cleared for SFT.
Turning it green would require either an audited action record for the
snapshot or weakening the gate; the second is forbidden and the first is not
this card's to write.

### Root cause

The control arm's stderr, verbatim (an `argparse` usage dump wrapping the
gate's own message):

```
usage: train_model.py [-h] [--train-dir TRAIN_DIR]
                      ... ~150 further option lines ...
                      [--allow-open-synthesis-feedback]
train_model.py: error: SFT blocked: open synthesis_feedback recommendations
without action/waiver record: ['id_collision', 'redundant_expansion',
'stale_output_contract']. Write
src/slm_training/resources/data/train/openui_verified_train_v1/synthesis_feedback_actions.json
after fixing the synthesizer (or an explicit waiver).
```

Two distinct things are happening, and only the second is a defect.

**1. The refusal itself — correct, not a bug.** The SFT data-governance gate
raises at
[`src/slm_training/autoresearch/hillclimb.py:1004`](../../src/slm_training/autoresearch/hillclimb.py)
(`assert_synthesis_feedback_cleared_for_sft`, the `still_open` branch) and is
re-raised through `parser.error(...)` at
[`scripts/train_model.py:1471`](../../scripts/train_model.py), which is why the
process exits **2** with a usage dump. The train snapshot
`openui_verified_train_v1` carries three open `synthesis_feedback`
recommendations — `id_collision`, `redundant_expansion`,
`stale_output_contract` — and no `synthesis_feedback_actions.json`. The gate
is behaving exactly as designed and is left untouched.

**2. The refusal is emitted untyped — the defect.** The generic non-zero-exit
path of `execute_commands`,
[`src/slm_training/autoresearch/engine.py:1400`](../../src/slm_training/autoresearch/engine.py)
(pre-fix line numbers), built its `ExperimentOutcome(status="failed", ...)`
with no `harness_signals` at all. Downstream, that has two consequences:

- `diagnose_outcome`
  ([`engine.py:1671`](../../src/slm_training/autoresearch/engine.py)) finds no
  reproduced signal and falls through to the
  `status == "failed" and not metrics` branch: `target="infrastructure"`,
  evidence = the entire ~4 KB argparse usage dump, recommended action
  "repair the failed harness stage and rerun the identical spec". The
  `p1-merged` status table shows that dump verbatim in its *Diagnostic
  Signals* row.
- `_primary_harness_family`
  ([`scripts/run_autotrain_continuous.py:13057`](../../scripts/run_autotrain_continuous.py))
  scans outcome artifacts for a signal with `reproduced_on_frozen_input` and,
  finding none, returns its `"model_build"` default. The `p1-merged`
  `cycle_handoff.json` therefore emits
  `repair_harness | owner=improve-openui-harnesses | harness_family=model_build`
  — the wrong owner. Model build cannot clear a train-data governance gate.

So the loop was asking the model-build owner to repair a train-data snapshot,
handing them an argparse dump as the diagnosis.

### The fix

`engine.stage_failure_signals(stdout, stderr)` classifies the *captured
output* of a failed stage into typed `HarnessSignalV1` evidence, and the
generic failure path now attaches it.

The classifier is **purely observational**: it reads what the stage actually
printed and never re-evaluates, re-implements, or relaxes any gate. The
stage-side gate remains the only authority on whether a stage may run. An
unrecognised failure stays untyped — the classifier never guesses.

Today it recognises one condition, the SFT refusal, and types it
`family="train_data"`, `code="sft_blocked_open_synthesis_feedback"`,
`evidence_uri=<the actions path the gate itself names>`.

Measured effect (`p1-fixed`, same arms, same snapshot):

| | `p1-merged` (before) | `p1-fixed` (after) |
| --- | --- | --- |
| `harness_signals` on the arm outcome | `[]` | `train_data / sft_blocked_open_synthesis_feedback` |
| Diagnosis target | `infrastructure` | `harness/train_data` |
| Diagnosis evidence | ~4 KB argparse usage dump | `sft_blocked_open_synthesis_feedback: src/.../openui_verified_train_v1/synthesis_feedback_actions.json` |
| `repair_harness` `harness_family` | `model_build` | `train_data` |
| Control arm exit | 2 | 2 (unchanged, correct) |

### Test

`tests/test_autoresearch/test_harness.py::test_sft_gate_refusal_routes_to_train_data_not_model_build`

It carries the c536–c543 control-arm stderr verbatim as a fixture, asserts the
**original** defect first (an untyped failed outcome still diagnoses as
`infrastructure` with the argparse dump as its evidence string), then asserts
the typed signal, the `harness/train_data` routing, and that an ordinary
traceback stays untyped.

### What this does *not* fix, and who owns it

- **The refusal itself.** `openui_verified_train_v1` is still not cleared for
  SFT, so the arm still exits 2 and no scoreboard is written. Clearing it is a
  **train-data / `synthesis-feedback`** decision: fix the certifier, or record
  an audited `synthesis_feedback_actions.json`. This card deliberately routes
  the repair to that owner rather than making the call itself. The three open
  recommendations all target the `corpus_certification` family of the source
  corpus `openui_verified_v1` (this snapshot is a root-family split of a
  certified corpus and synthesised nothing), so the upstream fixes are:
  namespace ids by source snapshot, dedupe at certification, and re-certify
  under `symbol_only/v2`.
- **The c536–c543 retry loop.** Already owned and already merged: card **P10**
  (`7ab7e45`, "heal executors for code/data classes; crash routing") added
  `HARNESS_CRASH_REASON_RE` and the `arm_exits`-aware crash/residual split in
  `src/slm_training/autoresearch/heal/classify.py`, whose docstring names
  "the misroute that let cycles c536..c543 drop their `repair_harness` blocker
  and continue". On the merged tree the blocker correctly stays hard, so the
  loop no longer thrashes. Note for P10: with `arm_exits {control: 2}` this
  refusal classifies as `BlockerClass.code` — a repo-internal harness bug —
  when it is really a data-governance park. `DATA_PREREQUISITE_MARKERS`
  already lists `synthesis_feedback.json`, but the reason string
  (`harness_failure:<arm>:experiment_failed`) never carries it, so the `data`
  route is unreachable for this condition. Now that the outcome carries a
  typed `train_data` signal, that routing can be sharpened.
- **RC1 `n=0` park.** Confirmed fixed elsewhere:
  `screening_smoke_n_for_policy(policy.v3 v15, arm_wall=70, suite_records=96)`
  returns `chosen_n=21`, verdict `insufficient_evidence`,
  `must_generate=False`.

## INT-1 — the SFT gate is cleared for the default corpus, and one data arm is withdrawn for measured eval leakage

Claim class: **data governance + harness change / fixture-demo**. No model was
trained and no gate, threshold or output contract was weakened. Follows the
`synthesis-feedback` law: act on the evidence, never on the gate.

### Why the loop was blocked

P1 established that every arm exited 2 because
`assert_synthesis_feedback_cleared_for_sft` correctly refused to train on
`openui_verified_train_v1`: its `synthesis_feedback.json` carries three open
recommendations (`stale_output_contract`, `id_collision`,
`redundant_expansion`) and no action record existed. The refusal was right;
the missing piece was the audited action record the gate is designed to read.

### What the evidence shows

Each recommendation prescribes an **admission-time** action, and the split
build already performs it. Re-measured from the committed artifacts rather
than asserted:

| Recommendation | Prescribed action | Measured on the admitted bucket |
| --- | --- | --- |
| `stale_output_contract` | reject, never patch, under `symbol_only/v2` | 90 `OutputContractError` rejections in `rejected.jsonl`; 0 admitted |
| `id_collision` | namespace ids by source snapshot | parent binds 303 ids to >1 distinct program (1,682 records / 700 ids, max multiplicity 7); admitted bucket is 1,083 records under 1,083 distinct ids, 0 collisions, scheme `root_id__content_digest8` |
| `redundant_expansion` | dedupe at admission | 175 duplicate pairs rejected; 0 exact prompt/program duplicate pairs among the 1,083 admitted |

`synthesis_feedback_actions.json` now records one `action_receipt` per code,
bound to the split-policy plan hash and the dataset manifest hash, with the
before/after counts above. It is not a waiver and not a prose note. It clears
SFT for `openui_verified_train_v1` only; re-certifying, namespacing and
deduping the **parent** corpus `openui_verified_v1` remain open against that
corpus's own feedback artifact.

### The leakage this surfaced

Checking whether the other candidate corpora were safe against the suites the
loop now scores produced a blocking finding. Overlap with
`e938_role_safe_all_targets_smoke96_v1`:

| Train corpus | Programs | Prompts | Root families |
| --- | --- | --- | --- |
| `openui_verified_train_v1` (policy default) | 0 | 0 | 0 |
| `hillclimb_strict_v2` (was the `data-strict` arm) | 6 | 3 | 16 |
| `wf_smoke_v2` (legacy control corpus) | 5 | 2 | 4 |

Against `held_out`, `hillclimb_strict_v2` overlaps 7 programs and 5 families.
Its own decontamination indexed `e938_..._v2`, `smoke6_v1` and `smoke24_v1`,
never `smoke96_v1` or `heldout24_v1`, so the overlap was never gated. An arm
trained on it would have scored the eval it memorised — a leakage win, not a
capability win, and precisely the kind of first "positive" that would have
made the recovery look successful while being worthless. Its synthesis
feedback is also uncleared for SFT (blocking-class `eval_leakage_source` on
three families, plus `dup_share` 0.55–0.93), so the arm could not have trained
at all.

Actions: `data-strict` is removed from `_SCREENING_ARM_BANK`;
`_LEAKED_TRAIN_VERSIONS` bars both corpora from any data arm; historical
ledger rows naming the withdrawn arm's corpus classify as
`data-leaked:hillclimb_strict_v2` so the ledger stays readable without the arm
becoming selectable (`wf_smoke_v2` is deliberately excluded from that
reverse-classification because it was a control corpus, never an arm).
`tests/test_scripts/test_screening_corpus_leakage.py` measures the overlap
from the committed data on every run: it fails if the default corpus ever
leaks, if any data arm's corpus leaks, or if a barred corpus becomes clean
(which should be a deliberate re-admission, not a silent one).

The certified default corpus stays: it is disjoint from every scored suite on
all three axes, which is what P7's root-family split was for.

## INT-2 — the certified train bucket was untrainable; v2 is the rebuild

Claim class: **data rebuild + harness change / fixture-demo**. No model was
trained to completion and nothing was promoted. No gate, threshold or output
contract was weakened; admission became stricter.

### What running the loop found that reading the corpus did not

With the SFT gate cleared (INT-1), one supervised cycle still failed both
arms. The traceback is not a park and not a crash in the driver:

```
File "src/slm_training/models/twotower.py", line 16583, in from_records
    assert_role_safe_output(record.openui, output_kind=record.target_kind)
ValueError: placeholder ':slot_0' in non-content property RadialChart.labels
```

`TwoTowerModel.from_records` applies the role-safe output contract to every
training record and raises on the first violation, which takes the whole arm
down. Certified records carry no `target_kind`, so the model resolves the
**document** contract — and admission had been applying a narrower one. A
first scan missed this entirely by passing the record's own (absent)
`target_kind`; under the contract the trainer actually resolves, **29 of the
1,083 admitted records** place a placeholder in a non-content property. Card
S2 had independently hit the same 29 from the other side: its n-gram builder
reported "1,083 records, 1,054 encoded, 29 refused by the codec".

The wider scan, under the document kind:

| Dataset | Records | Fail the trainer's contract |
| --- | --- | --- |
| `openui_verified_train_v1` | 1,083 | 29 — untrainable |
| `openui_verified_v1` (parent) | 1,682 | 888 |
| `wf_smoke_v2` | 101 | 0 |
| `smoke96_v1/smoke` | 96 | 5 |
| `smoke96_v1/held_out`, `heldout24_v1/held_out` | 24, 24 | 0 |

### The fix

`partition_certified_corpus` now applies **every** contract
`from_records` applies — `assert_no_template_semantic_labels`,
`assert_canonical_template_markers`, `assert_symbol_only_output` and
`assert_role_safe_output` — at a new `role_safety` admission stage, using the
record's own `target_kind` and falling back to the document kind the model
resolves. Checking only the role-safety one was not enough: the next cycle
failed on `template markers are opaque; semantic role labels are prohibited`
from a different contract, so the check is now kept in lock-step with the
trainer's list rather than extended one violation at a time. A failing
record is refused and written to `rejected.jsonl` like any other refusal —
nothing is dropped silently — and a `role_unsafe_output` recommendation
carries the evidence into `synthesis_feedback.json`.

Published datasets are immutable (`DataStore.publish` refuses an existing
destination) and the `synthesis-feedback` law says to rebuild under a new
version, so `openui_verified_train_v1` stays exactly as built and
**`openui_verified_train_v2`** is the rebuild: 893 train / 154 validation /
118 test, with 256 contract refusals across the three splits. Re-measured on
v2: **0 of 893 records fail any of the four trainer contracts**, 0 id
collisions, 0 exact duplicate pairs, and 0 program, prompt or family overlap
with any scored suite. v2's definition was corrected once, before any
measurement used it: the first cut checked one contract and admitted 1,054
records that still could not train. Its four recommendations carry
`action_receipt` records bound to the plan and manifest hashes.

`policy.v3` defaults, the driver's certified data-arm corpus, the certified
bucket id and the speculative n-gram table all move to v2. The n-gram table is rebuilt from v2
(893 sequences, 47,692 tokens, order 3, 493 contexts, 0 encode errors) and its
pinned branch points re-measured: `root = ` still picks `Stack(` at margin
15.000, and `root = Stack([` still picks `b1`, at margin 1.767 rather than
1.738. The doc paragraph, the artifact and the test moved together.

### Left open, deliberately

Five of the 96 `smoke96_v1/smoke` gold targets fail the same contract, so a
role-safe decoder can never reproduce them. They are not fixed here:
republishing an eval suite changes what every measurement is scored against,
which is a preregistration-sensitive change, and the screening verdict is a
**paired** per-record NLL comparison, so an unattainable target shifts both
arms equally and biases the absolute level rather than the delta. Recorded as
a measurement caveat and a successor card.

`hillclimb_strict_v2` remains uncleared for SFT (blocking-class
`eval_leakage_source`) and barred as a data arm for measured leakage (INT-1).
The upstream fixes to `openui_verified_v1` — re-certify under
`symbol_only/v2`, namespace ids, dedupe, and keep placeholders out of
non-content properties — remain open against that corpus's own feedback
artifact.

## INT-6 — three supervised cycles: the loop trains, measures and accumulates

Claim class: **fixture-scale recovery / fixture-demo**. No checkpoint was
promoted, synced or shipped, and no ship gate was evaluated. `--supervised`
runs exactly one bounded cycle, so this is three sequential supervised cycles
on one loop id from a fresh scratch root, each inside `MAX_RUN_MINUTES = 3`
per stage. No run timed out at the harness wall; the arm-level exits are
recorded below.

### What each cycle did

| Cycle | Regime | Control `smoke.eval_nll` | Records | Champion |
| --- | --- | --- | --- | --- |
| 1 | `isolate` (`no_climb_baseline_causal_ofat`) | 7.196 | 96 | none (cold start) |
| 2 | `climb` (`climb_champion_checkpoint_available`) | 4.569 | 96 | `baseline_seed`, 53 steps, 893 records |
| 3 | `climb`, `intent=retry_measurement` | 4.569 | 96 | unchanged (frozen replay) |

The primary is `smoke.eval_nll`, measured teacher-forced over all 96 records
of `e938_role_safe_all_targets_smoke96_v2` and persisted per record in
`eval_nll_records.json`, with `claim_class: diagnostic` on the scoreboard.

Cycle 2 is the first time this loop has ever carried state forward: the
control forked the seeded champion instead of starting from random init, and
its NLL fell from 7.196 to 4.569. Cycle 3 is a frozen replay of cycle 2's
measurement, which is why its number is identical — that is the retry path
working, not a second independent measurement.

For contrast, the committed history of the `continuous-openui-local` loop is
543 cycles with zero positives under the current eval key, 0 of 75 champion
promotions, and every arm in c536–c543 exiting 2.

### What is still incomplete, stated plainly

Every cycle above ends `measurement_complete: false` and `positive=False`.
Two things are missing, and neither is a quality verdict:

1. **No decoded-quality probe result.** The scoreboard carries `eval_nll` and
   nothing else: `document_n`, `completed_document_n` and
   `decode_timeout_count` are all `None`, which the completeness check reads
   as `invalid_counts`. Before INT-5 the probe decoded the whole 96-record
   suite and was killed at the wall after 80 records; with the probe bounded
   to 6 records the arm no longer dies, but the decoded evaluation now writes
   no progress file and no document counts at all. The knob is confirmed to
   reach the arm (`eval_limit: 6` in the executed experiment spec), so the
   remaining defect is downstream of the knob, in how a bounded certified
   suite is evaluated. That is the next card.
2. **No paired screening verdict.** Because the candidate arm's measurement is
   incomplete, no Wilcoxon over paired per-record NLL deltas is computed, so
   the loop reports neither a positive nor a decisive null. The NLL improvement
   in the table is a control-versus-itself accumulation across cycles, not an
   arm comparison, and must not be read as an experimental result.

The honest summary is that the loop now runs, trains, measures a primary on
96 records and carries a champion forward, and that the decoded-quality half
of the measurement is not yet complete.

## Test-type audit and the defects it found (2026-09-03)

The recovery above was reviewed for test coverage across the standard
types. The suite had atomic unit tests and integration tests and no instance
of chaos, fuzz/property, mutation, snapshot/characterization, contract, or
behavioral tests. Each type added found something, which is why they are
recorded here rather than counted as coverage.

### Defects found and fixed

| # | Found by | Defect |
|---|---|---|
| 1 | chaos | `run_playbooks` documented "a crashing playbook yields a `step_failed` receipt, never an exception" but guarded with `except Exception`, which does not catch `SystemExit`. A playbook driving an argparse-based module in process raises `SystemExit(2)` on a bad argv, so the invariant was false at the seam it was written for. `PLAYBOOK_FAULTS` now covers all five playbook boundaries; `KeyboardInterrupt` stays uncaught (the run cap interrupts with SIGINT, and kill is not evidence). |
| 2 | contract | `HealOutcome` carries both `verify_failed` and `postcondition_failed`; `_record_pass_outcome` mapped only the first, so `postcondition_failed` — the `data_rebuild` playbook's verdict when its record-count postcondition does not hold — fell through the `.get` default to the generic `heal_attempted`. The outcome that exists to make a no-op data heal visible was itself being flattened. `_PASS_OUTCOME_BY_HEAL_OUTCOME` is now total over `HealOutcome`. |
| 3 | behavior | `_park_screening_n_deficit` hard-coded "screening suite_volume binds" and a `rebuild_data` action for every empty range. Under a `wall_budget` bind that asks the synthesis owner for work that cannot lift the block: they publish records, the range stays empty, and the next cycle parks again. Actions now follow the report's `binding_constraints`. |
| 4 | full-suite run | Five tests were already red on `main` (differential parser audit firing on unresolved references; three catalog families with no mixture weight; an adapter allowlist grading text rather than paths; two serving tests asserting named prompt markers the opaque-marker contract forbids). Repaired without weakening any gate. |

### Recorded findings, no change made

- **`assert_symbol_only_output` is shadowed by `assert_role_safe_output`** on
  every *document*-kind input we could construct. It is independently
  observable only at `target_kind="lexical"`, where role safety abstains —
  and admission resolves `record.target_kind or "document"`, so a lexical
  record is guarded by that contract alone. Both contracts stay; the mutation
  suite pins the lexical case that makes the second one load-bearing.

- **Named markers in a live serving prompt are refused, not canonicalized.**
  `canonicalize_example_template_markers` rewrites every surface of a
  *persisted* record to opaque ordinals. Doing the same for a live request —
  prompt, design context and targets, then mapping back after decode — would
  let `Placeholders: :promo.title` through again. That is a serving feature,
  not a repair, and is left to the serving owner. The refusal is a
  `ValueError`, which the route already answers as 400, and its message now
  names the caller's inventory and the remedy.

### The process gap that let 4 happen

`verify_merge_ready --fast` is a static profile and never runs the suite. The
full profile ends with `scripts.check_changed --changed-tests-only`, and CI's
`python` job runs the same selection. That selection has an inversion:

```
['…/climb_policy.py', '…/policy.v3.json']            -> tests/test_autoresearch
['…/climb_policy.py', '…/policy.v3.json', <a test>]  -> just <a test>
```

`select_changed_tests` returns the changed *test files* whenever there are
any, and only otherwise falls back to the suites owning the changed *source*.
So adding a test to a change **shrinks** the tests the gate runs. That is how
a merge changing a policy resource and its own tests reported green while
eight tests in untouched files went red, and part of why five more sat red on
`main`.

One piece is fixed: a changed *global* test file (`tests/conftest.py`,
`tests/casefiles.py`, `pyproject.toml`) now always wins, because it means
every suite is in scope and the changed-file shortcut must not suppress it.

The general fix — a union, making the selection monotone — is **not** applied.
It is the correct semantics, but measured on a representative diff it selects
`tests/test_data`, `tests/test_web`, `tests/test_harnesses/train_data` and
more, each of which alone exceeds `MAX_RUN_MINUTES` on this box, and CI runs
the same selection under a budget that is disabled for cost
(`docs/design/ci-minutes-and-speed-plan-20260806.md`). Trading local-commit
latency and CI minutes for that coverage is the owner's decision, not a
repair to make silently. `tests/test_scripts/test_check_changed.py` pins the
current behaviour under a test named for the non-monotonicity so the choice
stays visible.

Until it is decided, the operational rule is: **run the owning suites for what
you changed before merging, not only the tests you edited.**

## The committed demo checkpoint, and why it still cannot be regenerated (2026-09-03)

`src/slm_training/resources/checkpoints/playground_demo/last.pt` predates
`symbol_only/v2`, so `TwoTowerModel.from_checkpoint` refuses it:

```
OutputContractError: checkpoint output contract v0 is incompatible with
required symbol_only/v2; retrain from symbol-only targets
```

The gate is right. Three tools default to that checkpoint and are blocked by
it: `scripts/run_perf_matrix.py`, `scripts/run_discrete_plan_pareto.py`, and
`scripts/export_playground_onnx.py` — plus the S5/S6/S12/S14 audit cards,
which recorded the same blocker.

### Fixed: the regeneration path

`scripts/bootstrap_playground.py`, the documented way to rebuild it
(`docs/MODEL_CARD.md`), could not run either. Its `DEMO_RECORDS` spelled
semantic markers (`:hero.title`, `:cta.label`, `:pricing.cta`) that the
opaque-marker contract forbids, and the records declared no `placeholders`
list. Both are fixed: markers are opaque and contiguous from `:slot_0`, the
inventory is derived from the serialized program so the two can never
disagree, and `_assert_trainable` now applies the trainer's four contracts
before spending any training time, naming the offending record.

Verified end to end: `python -m scripts.bootstrap_playground --output <scratch>
--steps 200` trains in seconds (loss 1.85 → 0.007) and the result loads
through `TwoTowerModel.from_checkpoint` under `symbol_only/v2`.
`tests/test_scripts/test_bootstrap_playground.py` keeps the corpus honest.

### Not fixed: the ONNX serving path has drifted from the trainer

Regenerating the committed artifacts was attempted and **reverted**. The
checkpoint and its ONNX sidecars rebuild cleanly (`onnx` is a declared `web`
extra, absent from this environment until installed), but the result breaks
`tests/test_web/test_onnx_inference.py`, 5 tests:

```
ValueError: tokenizer version mismatch: file has v5, expected v2
  — retrain or rebuild checkpoint
```

`models/onnx_inference.py` loads the tokenizer sidecar with
`OpenUITokenizer.load`, whose `TOKENIZER_VERSION` is **2** (72 tokens in the
committed artifact). The current `TwoTowerModel` writes a **v5** tokenizer
(569 tokens). So the ONNX serving path cannot consume *any* checkpoint the
current trainer produces: it works today only because the committed artifacts
are frozen at the old tokenizer, alongside the old output contract.

That makes the committed demo checkpoint self-consistent but jointly stale,
and there is no in-between: replacing `last.pt` alone leaves the ONNX
sidecars serving a different model, and replacing all of them breaks ONNX
inference. Porting `onnx_inference.py` to the current tokenizer is a serving
migration with its own validation, not a boy-scout repair, so it is recorded
here rather than half-done.

Until it lands, `run_perf_matrix`, `run_discrete_plan_pareto` and the decode
audit cards need a checkpoint trained on this box (the
`slm322_ap027_scratch_v1` precedent in `docs/MODEL_CARD.md`), not the
committed fixture.

## `forced_token_fraction` under-reported deterministic forcing (2026-09-03)

The S12 commit-authority profile recorded that `forced_tokens` "stayed 0 in
every run while `semantic_singleton_bypasses` was 18–27, because the bypasses
landed in constrained LTR repair, which increments
`forced_row_tokens_without_forward` only", and flagged it for S4's derived
ratio. Confirmed and fixed.

`DecodeStats.forced_token_fraction` is documented as *"the share of the canvas
that deterministic forcing (I2 singleton bypass and forced spans) wrote
without consulting the model"*. Four of the six sites that commit a singleton
bypass called `_record_exact_bypass` and incremented
`forced_row_tokens_without_forward` without touching `forced_tokens`
(`twotower.py` 5850, 12644, 13528, 13795). The ratio therefore attributed
forced tokens to the model.

Measured on a minimal grammar-determined canvas (`root = Separator()`, 8
committed tokens, all of them proven forced):

| | `forced_tokens` | `forced_token_fraction` |
|---|---|---|
| before | 5 | 0.625 |
| after | 8 | 1.000 |

Telemetry only — no decode decision, ordering or output changes; each added
line counts an event the adjacent line already counted under another name.
`verify_decode_invariants` stays green and the 371 twotower/forcing tests pass.

`tests/test_models/test_forced_token_fraction.py` asserts the exact value
rather than `> 0`, because `> 0` passes on the broken code: five of eight is
still positive. That is how a telemetry defect survives a test suite, and it
is worth stating as a rule — **a ratio is pinned by its value, never by its
sign.**
