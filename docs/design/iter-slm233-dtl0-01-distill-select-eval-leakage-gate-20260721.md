# SLM-233 (DTL0-01): distillation trace-selection eval-holdout leakage gate stress test (slm233-distill-select-eval-leakage-gate-20260721)

**Matrix set:** `slm233_distill_select_eval_leakage_gate`
**Version:** `dtl0-01-v1`
**Status:** fixture
**Claim class:** wiring
**Fixture suites:** smoke, held_out, adversarial, ood
**Gate hash:** `59e3e5cb98d804d3...`
**Control selection rate:** 1.00
**Disposition:** gap_confirmed — 4/4 eval-suite arms (held_out_suite_traces, adversarial_suite_traces, ood_suite_traces, smoke_suite_traces) were selected by the real select_traces / filter_traces pipeline at the same rate as the train-prompt control (1.00), and a static audit confirms the P2 selection code path never reads meta.source_suite or any other eval-provenance field. The documented 'selection data stays disjoint from frozen evals; never train on held-out benchmark traces' invariant has no enforcement in select_traces.

## Hypothesis

The real select_traces / filter_traces pipeline, exercised exactly as scripts/self_distill.py select calls them with default flags, has no discrimination against traces whose meta.source_suite marks them as coming from a frozen eval suite: it selects traces rolled out over the committed held_out/adversarial/ood/smoke eval suites (the CLI default for P1 trace collection) into the self-distillation training corpus at the same rate as traces rolled out over ordinary train prompts, with no error, no warning, and no different treatment.

## Falsifier

select_traces / filter_traces (or scripts/self_distill.py select around them) reject, warn on, drop, or otherwise select eval-suite-sourced traces at a different rate than ordinary train-sourced traces; or the distill.select module already reads meta.source_suite (or any other eval-provenance field).

## Static source-suite-provenance audit

| module | references eval-suite provenance |
| --- | --- |
| slm_training.harnesses.distill (__init__) | False |
| slm_training.harnesses.distill.select | False |

## Honest caveats

- Fixture/wiring evidence only: no checkpoint, GPU run, self-distillation SFT step, or ship-gate claim is made or implied.
- This exercises the real, unmodified build_test_data, DecodeTraceRecorder.finalize, TraceStore.append, and select_traces functions against a tiny (16-record) committed fixture corpus, not a production-scale evaluation or training run.
- Traces are constructed directly via the real DecodeTraceRecorder / TraceStore API rather than by running an actual MaskGIT rollout through a checkpoint (that would need a real trained model and would not change what select_traces sees: meta.source_suite, labels, and reward are the same fields collect_trajectories.py writes). The synthesized final text is a minimally whitespace-mutated copy of each record's gold OpenUI (grammar-valid, not exact_gold) standing in for a plausible accepted decode; it is not a claim about real model output quality.
- Whether training a real self-distillation SFT step on eval-suite-sourced traces actually harms downstream ship-gate scores (as opposed to only being an unenforced provenance gap) is not measured here; this harness is about the P2 selection *pipeline*, not distillation training dynamics.
- The static source-audit only inspects the distill.select and distill package modules reachable from self_distill.py select's default path; it does not prove the absence of leakage checks anywhere else in the repository, and does not inspect P1 collect_trajectories.py itself (which is the CLI that writes meta.source_suite in the first place, not the selection stage under test).
- distill.select does import slm_training.data.leakage.fingerprint_openui_structure, but only to build the structural-shape component of a trace's stratification key (coverage-over-score sampling), never to compare against an eval-suite fingerprint set; the static audit's needle list deliberately excludes the bare substring 'leakage' for this reason and checks for source_suite/held_out/eval_suite/test_dir instead.

## Per-arm results

| arm | source_suite | n traces | n selected | selection rate | gameable | control |
| --- | --- | --- | --- | --- | --- | --- |
| train_prompts_control | None | 20 | 20 | 1.00 | False | True |
| held_out_suite_traces | held_out | 5 | 5 | 1.00 | True | False |
| adversarial_suite_traces | adversarial | 4 | 4 | 1.00 | True | False |
| ood_suite_traces | ood | 4 | 4 | 1.00 | True | False |
| smoke_suite_traces | smoke | 3 | 3 | 1.00 | True | False |

## Arm descriptions

- **train_prompts_control**: Traces rolled out over ordinary train prompts (src/slm_training/resources/train_seeds.jsonl), as if collected via `collect_trajectories.py --records train_seeds.jsonl` (no --test-dir, so source_suite=None). Negative control: should select into the corpus normally.
- **held_out_suite_traces**: Traces rolled out over the real, freshly-built held_out eval suite, as if collected via `collect_trajectories.py --test-dir <dir> --suite held_out` (the CLI default).
- **adversarial_suite_traces**: Traces rolled out over the real, freshly-built adversarial eval suite, as if collected via `collect_trajectories.py --test-dir <dir> --suite adversarial`.
- **ood_suite_traces**: Traces rolled out over the real, freshly-built ood eval suite, as if collected via `collect_trajectories.py --test-dir <dir> --suite ood`.
- **smoke_suite_traces**: Traces rolled out over the real, freshly-built smoke eval suite, as if collected via `collect_trajectories.py --test-dir <dir> --suite smoke`.

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `select_traces`, `filter_traces`, `collect_trajectories.py`, or `self_distill.py`, does not run self-distillation SFT, and makes no ship or gate claim. It documents a concrete gap between the documented 'selection data stays disjoint from frozen evals; never train on held-out benchmark traces' invariant and the actual P2 selection code path, as a candidate for a future, separately reviewed hardening change (never implemented here).

## Reproducibility

```bash
python -m scripts.run_slm233_distill_select_eval_leakage_gate --mode plan-only
python -m scripts.run_slm233_distill_select_eval_leakage_gate --mode fixture
```
