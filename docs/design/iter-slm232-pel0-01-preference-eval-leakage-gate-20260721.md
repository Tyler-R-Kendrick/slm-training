# SLM-232 (PEL0-01): preference build-pairs eval-holdout leakage gate stress test (slm232-preference-eval-leakage-gate-20260721)

**Matrix set:** `slm232_preference_eval_leakage_gate`
**Version:** `pel0-01-v1`
**Status:** fixture
**Claim class:** wiring
**Fixture suites:** smoke, held_out, adversarial, ood
**Gate hash:** `2e7b684810ad9078...`
**Disposition:** gap_confirmed — 4/4 eval-suite arms (held_out_as_train_records, adversarial_as_train_records, ood_as_train_records, smoke_as_train_records) were accepted by the real collect_pairs_with_generator / write_pairs pipeline and produced valid preference pairs while 100% fingerprint-identical to the eval suite they were drawn from, and a static audit confirms the preference build-pairs code path never references the leakage/disjointness module test-data already trusts for the opposite direction. The documented 'never train on eval-feedback holdouts' invariant has no enforcement in build-pairs.

## Hypothesis

The real collect_pairs_with_generator / write_pairs pipeline, exercised exactly as scripts/train_preference.py build-pairs calls them with no --from-checkpoint, has no leakage/disjointness enforcement against eval suites: it accepts records drawn verbatim from a freshly-built, genuine held_out/adversarial/ood/smoke eval suite (built by the real, unmodified build_test_data pipeline) and emits valid preference pairs with no error, no warning, and no eval-provenance flag -- exactly as if those records were ordinary train data -- while the equivalent train-side disjointness check (find_leakage / load_train_fingerprints) that build_test_data already enforces would flag every one of those records as 100% fingerprint-identical to the eval suite it came from.

## Falsifier

collect_pairs_with_generator / write_pairs (or scripts/train_preference.py build-pairs around them) reject, warn on, or otherwise flag eval-suite-sourced records differently from ordinary train records; or the preference package already imports/calls the leakage/disjointness module.

## Static leakage-import audit

| module | references leakage module |
| --- | --- |
| slm_training.harnesses.preference (__init__) | False |
| slm_training.harnesses.preference.train | False |

## Honest caveats

- Fixture/wiring evidence only: no checkpoint, GPU run, preference training step, or ship-gate claim is made or implied.
- This exercises the real, unmodified build_test_data, collect_pairs_with_generator, write_pairs, find_leakage, and load_train_fingerprints functions against a tiny (16-record) committed fixture corpus, not a production-scale evaluation or training run.
- The fingerprint-match candidate check (reusing load_train_fingerprints / find_leakage, pointed at the eval suite instead of a train manifest) is an illustrative diagnostic only. It is not implemented in the preference harness or scripts/train_preference.py, not proposed as the correct fix, and passing/failing it makes no gate or promotion claim.
- Whether training a real preference model on eval-suite-identical pairs actually harms downstream ship-gate scores (as opposed to only being an unenforced provenance gap) is not measured here; this harness is about the build-pairs *pipeline*, not preference training dynamics.
- The default (no --from-checkpoint) build-pairs candidate generator is reproduced inline (gold + soft-corrupt reject) to match the CLI exactly; it is not imported from scripts/train_preference.py because that module is a __main__ script, not an importable library function.
- The static source-audit only inspects the preference package modules reachable from build-pairs' default path; it does not prove the absence of leakage checks anywhere else in the repository.

## Per-arm results

| arm | n records | build-pairs succeeded | pairs written | match rate vs. source-suite fingerprints | gameable | control |
| --- | --- | --- | --- | --- | --- | --- |
| train_seeds_control | 20 | True | 20 | 0.00 | False | True |
| held_out_as_train_records | 5 | True | 5 | 1.00 | True | False |
| adversarial_as_train_records | 4 | True | 4 | 1.00 | True | False |
| ood_as_train_records | 4 | True | 4 | 1.00 | True | False |
| smoke_as_train_records | 3 | True | 3 | 1.00 | True | False |

## Arm descriptions

- **train_seeds_control**: Ordinary train records (src/slm_training/resources/train_seeds.jsonl) fed to build-pairs as intended. Negative control: should build successfully and NOT fingerprint-match the held_out eval suite.
- **held_out_as_train_records**: The real, freshly-built held_out eval suite fed to build-pairs as --train-records.
- **adversarial_as_train_records**: The real, freshly-built adversarial eval suite fed to build-pairs as --train-records.
- **ood_as_train_records**: The real, freshly-built ood eval suite fed to build-pairs as --train-records.
- **smoke_as_train_records**: The real, freshly-built smoke eval suite fed to build-pairs as --train-records.

## No-go for promotion

This report is wiring/fixture evidence only. It does not change `collect_pairs_with_generator`, `write_pairs`, `build_test_data`, or `find_leakage`, does not train a preference model, and makes no ship or gate claim. It documents a concrete gap between the documented 'never train on eval-feedback holdouts' invariant and the actual preference build-pairs code path, as a candidate for a future, separately reviewed hardening change (never implemented here).

## Reproducibility

```bash
python -m scripts.run_slm232_preference_eval_leakage_gate --mode plan-only
python -m scripts.run_slm232_preference_eval_leakage_gate --mode fixture
```
