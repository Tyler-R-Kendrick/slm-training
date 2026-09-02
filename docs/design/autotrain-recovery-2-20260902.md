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
