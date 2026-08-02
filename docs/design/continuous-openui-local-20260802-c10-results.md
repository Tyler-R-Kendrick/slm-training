# Continuous autotrain cycle 10 results (2026-08-02, loop `continuous-openui-local`)

**Honesty:** `fixture_or_scratch` only. **Not a ship claim.**

| Field | Value |
| --- | --- |
| Loop | `continuous-openui-local` |
| Campaign | `continuous-loop-20260802-continuous-openui-local-8c0b60dd-c10` |
| Cycle intent | **harness/data improvement** (test_data family) — not a model screen, unlike cycles 1-9 |
| Upstream / integration | `b8188a49` / `2d9a137b` |
| Device | CPU |

This cycle had two parts: (1) grow the published smoke/held_out eval suite
past n=3/5, and (2) use it for one diagnostic screening cycle to test cycle
9's open question: does a larger suite reveal a non-null signal for a
structural aux-loss lever, or is it still bit-identical?

## Part 1 — published a larger eval snapshot

Read `.claude/skills/autotrain/references/test-data.md` in full first.
`slm data build-test` only draws `smoke`/`held_out` from the hand-authored
fixture pool `src/slm_training/resources/test_seeds.jsonl` (`--target-records`
only expands `rico_held`, per `TestDataConfig.rico_suites`) — so growing
`smoke`/`held_out` means hand-authoring new fixture records, not a CLI flag.

- Extended `test_seeds.jsonl` **purely additively**: 16 → 35 records. No
  existing record touched. `smoke` +9 (3→12), `held_out` +10 (5→15);
  `adversarial`/`ood` left at 4/4.
- New records exercise components the original 16 seeds never touched:
  `RadioGroup`/`RadioItem`, `Select`/`SelectItem`, `CheckBoxGroup`/
  `CheckBoxItem`, `TextArea`, `DatePicker`, `Steps`/`StepsItem`, `Table`/
  `Col`, `Accordion`/`AccordionItem`, `Carousel`, `TagBlock`, `CodeBlock`,
  `MarkDownRenderer`, several composed into `Form`/`FormControl`/`Card` for
  `held_out`.
- Every new record: parses via `validate()`, passes
  `assert_symbol_only_output()`, and its structural fingerprint collides
  with none of: the other 18 new records, the existing 16 test seeds,
  `train_seeds.jsonl`'s 21 hand-authored train fixtures, or `wf_smoke_v2`'s
  48 published train `structure_fingerprints` (the `--train-manifest` used
  for the real build). **`error_count=0`, `leakage_rejected=0`** out of 35
  kept records — no rejection to report, no gate weakened.
- Published to `src/slm_training/resources/data/eval/test_data_scaleup_v1/`
  (same layout as `e763`/`e827`/`e842`/`e938`) via:
  ```
  python -m scripts.build_test_data --source fixture \
    --seed-path src/slm_training/resources/test_seeds.jsonl \
    --train-manifest src/slm_training/resources/data/train/wf_smoke_v2/manifest.json \
    --suites smoke,held_out,adversarial,ood \
    --version test_data_scaleup_v1 \
    --output-root src/slm_training/resources/data/eval --register-lineage
  ```
- Bumped `data.test_build` v6→v7 in `versions.json` (real content change, not
  a no-bump). `verify_version_stamps --check` clean.
- `npm ci` was required in `src/apps/openui_bridge` — the bridge wasn't
  installed at cycle start, so `tests/test_harnesses/test_data` and
  `tests/test_integration/test_ship_disjoint.py` were bridge-skipped; all 12
  pass after install, plus the dedicated symbol-only/locked-manifest tests.
- Committed as `2d9a137` (`feat(data): publish test_data_scaleup_v1`).

**Result: `test_data_scaleup_v1`, `smoke=12` / `held_out=15` / `adversarial=4`
/ `ood=4` — a genuine, quality-gate-clean, meaningfully larger snapshot.**

## Part 2 — wiring it into the continuous driver

`scripts/run_autotrain_continuous.py` has **no `--eval-version` flag**; it
always calls `slm_training.autoresearch.engine.default_eval_version()`,
which walks a hardcoded `_DEFAULT_EVAL_VERSION_CANDIDATES` tuple. The natural
fix — prepend `test_data_scaleup_v1` to that tuple in `engine.py` — was
**attempted but not committed**: the repo's pre-commit hook
(`.githooks/check-changed`) maps any diff to `engine.py` onto the full
`harness.autoresearch.experiment_campaign` test component, which includes
`tests/test_autoresearch/test_climb_policy.py::test_continuous_classify_positive_entry`.
That test was confirmed **already failing on the untouched baseline commit
`38b2f0c`** (reproduced in isolation with the change stashed out) — a
pre-existing bug where `_classify_positive`'s "fail closed on incomplete
evidence" guard (a recently-merged upstream law) now unconditionally rejects
positivity whenever `scoreboard.json` is absent, but the test fixture never
writes one. This is unrelated to this cycle's change and out of the
`test_data`-family scope declared for this cycle to fix (and the repo's own
rule keeps ordinary pytest files read-only for agents, unlike the mirrored
`test_cases` JSON fixtures).

Rather than force the commit through or skip the hook (never done without
explicit user request), the `engine.py`/`versions.json` edit was **reverted**
and a **non-tracked, scratch launcher script** was used instead: it
monkeypatches `engine._DEFAULT_EVAL_VERSION_CANDIDATES` in-process
immediately before calling `scripts.run_autotrain_continuous.main()`. Verified
working via the campaign's own `matrix-proposal.json`:
`hypotheses[].experiment.knobs.eval_version == "test_data_scaleup_v1"`.
**Recommendation for cycle 11+:** fix
`test_continuous_classify_positive_entry` (or narrow the hook's
`engine.py → test_climb_policy.py` mapping) so this registration can land as
a real, durable commit instead of a per-session workaround.

## Part 3 — the screening attempt

Driver's hypothesizer selected `component-structure` (rank-1 recommendation
carried over from cycle 9's next-run priorities), steps=21, seed=100010,
both arms 1,913,789 params.

| Arm | Levers | steps | eval_version | Outcome |
| --- | --- | ---: | --- | --- |
| c10-control | component-structure profile, all aux weights 0 | 21 | `test_data_scaleup_v1` | **measurement_incomplete** |
| c10-component-structure | component-structure profile, plan/edge weight 1.0 | 21 | `test_data_scaleup_v1` | **measurement_incomplete** |

Both arms' `evaluate_model --suites smoke` stage was interrupted by the
harness's fixed per-arm wall-time budget: `_arm_wall_minutes = (
MAX_HARNESS_WALL_SECONDS(155s) - HARNESS_FINALIZATION_RESERVE_SECONDS(15s)
) / 2 = 70s`, covering **both** `train_model` and `evaluate_model` for that
arm together. That 70s budget was implicitly sized against `n=3` smoke
(cycles 6-9 finished whole 2-arm campaigns in 19-31s total). At `n=12`, each
arm's `decode_progress.json` shows `status="interrupted"`,
`processed_record_n=2` — only 2 of 12 smoke records decoded before the
stage was cut off.

`SDLC Phase A`: `positive=false`, `stack_layer=false`,
`action=no_stack_layer_non_positive`. Reasons (from `sdlc_delivery.json`):

1. `measurement_incomplete:c10-control:missing_scoreboard`
2. `measurement_incomplete:c10-component-structure:missing_scoreboard`
3. `wall_timeout:efb86885c7...` / `wall_timeout:75d3cfe7d8...`
4. `primary_metric_unavailable`

The driver's own next-run priority (rank 1, confidence 0.95): "measurement
is incomplete; replay the exact frozen control and candidate before testing
a new hypothesis" (`action.kind=retry_measurement`, `owner=autotrain`).

## Part 4 — bounded reduced-scope retry attempt

Per this cycle's mandate ("reduce scope … and retry once if that's a quick,
well-justified adjustment"), a **local-only** (never committed,
`outputs/data/eval/`) `smoke=6` probe snapshot
(`test_data_scaleup_v1_smoke6_probe`: the original 3 smoke records + 3 of the
simplest new single-component additions — `TextArea`, `DatePicker`, `Tag`)
was built and a retry was attempted. It was **blocked before reaching the
campaign at all**: `run_autotrain_continuous.py` refuses to start a successor
cycle while a predecessor campaign has an unacknowledged `document` action
(`RuntimeError: predecessor … has unacknowledged actions: 1:document`) — the
repo's own Iron Law enforcement firing correctly, requiring this
documentation pair plus an `ack-action` receipt before any cycle 11 can run.
The `smoke6` probe is left built and validated
(`error_count=0`, `leakage_rejected=0`) under `outputs/data/eval/` for
cycle 11 to consume directly if the orchestrating session wants to continue
the reduced-scope retry.

## The key question this cycle answers

Cycle 9 posed two possible outcomes for a larger suite: reveals a signal, or
stays bit-identical. **Neither happened.** At `n=12` the larger suite could
not complete a comparable decode pass **at all** within the existing
per-cycle wall-time budget — only 2 of 12 smoke records decoded per arm in
~70s. This reframes, rather than resolves, cycles 6-9's bottleneck finding:
`n=3/5` genuinely was too small to detect these levers' effects, but simply
publishing a bigger fixture snapshot is **not sufficient on its own** — the
continuous driver's fixed per-arm wall budget (sized for `n=3`) does not
scale to `n=12` on CPU-bound constrained decode, especially for records
exercising components (`RadioGroup`/`Select`/`Table`/etc.) the tiny
`wf_smoke_v2` scratch model has had far less training signal on, which
appears to push per-record decode close to the 24s per-record
`--decode-timeout-seconds` ceiling. A real fix needs either a larger/
role-scaled per-arm wall budget for eval-scale screens, sharding the smoke
suite's decode across cycles, or a shorter per-record decode timeout for
screening role — none of which were attempted this cycle (out of the
`test_data`-family scope; flagged below for cycle 11+).

## Next-run priorities

1. **infrastructure (driver's own rank 1):** replay the exact frozen c10
   control/candidate before testing a new hypothesis.
2. **infrastructure:** either raise `_arm_wall_minutes` for eval-scale
   screening cycles specifically, or retry with the already-built
   `outputs/data/eval/test_data_scaleup_v1_smoke6_probe` (`smoke=6`) — a
   quick, bounded next step that doesn't require more fixture authoring.
3. **infrastructure:** fix `test_continuous_classify_positive_entry` (write
   a `scoreboard.json` in its fixture, matching the "fail closed on
   incomplete evidence" contract it predates) so
   `_DEFAULT_EVAL_VERSION_CANDIDATES` can be durably updated in a normal
   commit instead of a per-session monkeypatch.
4. **model:** `component-structure` remains untested (this cycle's attempt
   produced no comparable metrics); re-screen once eval-scale measurement
   completes.
5. **infrastructure:** soft ship-gate fails / wall timeouts on a fixture
   `n` scale-up never stop the continuous loop — this cycle's `positive:
   false` classification is a legitimate, informative infra finding, not
   evidence against the levers under test.

## Screening-bank assessment

This cycle does not add a fifth bit-identical null to the cycle 6-9 pattern
— it is a distinct outcome (`measurement_incomplete`, not a completed null
comparison), and should not be read as "component-structure also nulls."
The `component-structure` arm is **not exhausted**; it simply has no
completed measurement yet at any suite scale this session.

## Artifacts

- Eval snapshot: `src/slm_training/resources/data/eval/test_data_scaleup_v1/`
  (committed `2d9a137`)
- `smoke=6` retry probe (uncommitted, local-only):
  `outputs/data/eval/test_data_scaleup_v1_smoke6_probe/`
- Campaign: `outputs/autoresearch/continuous-loop-20260802-continuous-openui-local-8c0b60dd-c10/`
- Runs: `.../runs/c20260802-continuous-openui-local-8c0b60dd-c10-{control,component-structure}/`
- Handoff: `.../cycle_handoff.json`
- SDLC delivery: `.../sdlc_delivery.json`
- JSON twin: `continuous-openui-local-20260802-c10-results.json`
- Predecessor: [cycle 9 results](continuous-openui-local-20260802-c9-results.md) (`binder-topology`, bit-identical null at 2x steps — the cycle whose open question this one attempted to close)
