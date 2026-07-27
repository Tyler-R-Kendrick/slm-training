# SLM-430: wire the locked-manifest digest check into real promotion CLIs

SLM-306 (`docs/design/slm306-locked-manifest-digest-verification.md`) gave
`load_campaign_governance` / `evaluate_promotion` / `register_promoted_checkpoint`
(`src/slm_training/harnesses/experiments/promotion.py`) an optional
`locked_manifest_path: Path | None = None` parameter. When supplied, a
campaign's self-reported `locked_eval_manifest_sha256` is independently
re-derived from the real bytes of the committed locked manifest
(`src/slm_training/data/locked_eval_manifest.canonical_manifest_path()`)
instead of only being trusted as a free-form hex string. That parameter
defaulted to `None` everywhere, and SLM-306's own writeup flagged the gap it
left open:

> Wiring `canonical_manifest_path()` in as the default for the real promotion
> CLI entrypoints (`scripts/resume_climb.py`, `scripts/run_scaling_ladder.py`,
> and `ModelBuildConfig.register_promoted` in `train_loop.py`) is a natural,
> still-narrow follow-up ... it was intentionally left out here to avoid
> changing default gate behavior without a live promotion run to validate
> against.

Reading those three entrypoints confirmed the gap was worse than "off by
default": none of them ever passed `locked_manifest_path` at all, by default
*or* otherwise. There was no way to request the stronger, content-addressed
check from any real promotion run -- the opt-in mechanism existed only in
library code with no reachable caller.

This change is distinct from SLM-306: SLM-306 built the verification
primitive and threaded the parameter through the library functions. This
closes the next narrow gap -- making that primitive *reachable* from the
three real entrypoints -- without doing what SLM-306's own note cautioned
against (flipping the *default* for every existing promotion run without a
live run to validate against). No GPU or live promotion run is available in
this CPU-only sandbox, so the safer, still-valuable increment is: add an
explicit, default-off opt-in at each entrypoint, verified with fixture/CPU
evidence against the real committed manifest.

## What changed

- `src/slm_training/harnesses/model_build/config.py`: `ModelBuildConfig`
  gains `verify_locked_manifest_digest: bool = False`.
- `src/slm_training/harnesses/model_build/train_loop.py`: `train()`'s
  `register_promoted` path computes
  `locked_manifest_path = canonical_manifest_path() if config.verify_locked_manifest_digest else None`
  once, and passes it to `load_campaign_governance`, `evaluate_promotion`, and
  `register_promoted_checkpoint`.
- `scripts/resume_climb.py` and `scripts/run_scaling_ladder.py`: both gain a
  `--verify-locked-manifest-digest` flag (`action="store_true"`, default
  `False`) with identical semantics, threaded through the same three calls
  (`run_scaling_ladder.py` threads it via its existing `governance_kwargs`
  dict, shared by `evaluate_promotion` and `register_promoted_checkpoint`).

In every case, omitting the flag reproduces the exact prior behavior
(`locked_manifest_path=None`, self-consistency-only check) -- this is
additive, matching SLM-306's own non-breaking posture. Setting the flag makes
the entrypoint fail closed (`campaign governance validation failed:
locked_eval_manifest_digest_unverified_on_disk` from `load_campaign_governance`,
or the `campaign_governance` check block failing inside `evaluate_promotion`)
if the campaign's declared `locked_eval_manifest_sha256` does not match the
real, on-disk `canonical_manifest_path()` bytes.

## Evidence (local CPU, fixture/audit scale)

`docs/design/slm430-promotion-cli-locked-manifest-digest-opt-in-20260727.json`
records the canonical manifest's real digest/row-count (recomputed via
`load_locked_manifest_payload`, matching SLM-306's own recorded value) and the
new test module's pass/fail table.

`tests/test_harnesses/experiments/test_locked_manifest_digest_threading.py`
(7 tests, all passing) proves, for each of the three entrypoints
(`train()`, `scripts/resume_climb.py::main`, `scripts/run_scaling_ladder.py::main`):

| entrypoint | flag absent | flag present |
| --- | --- | --- |
| `train()` (`ModelBuildConfig.register_promoted`) | `locked_manifest_path is None` | `locked_manifest_path == canonical_manifest_path()` |
| `resume_climb.main` | `locked_manifest_path is None` | `locked_manifest_path == canonical_manifest_path()` |
| `run_scaling_ladder.main` | `locked_manifest_path is None` | `locked_manifest_path == canonical_manifest_path()` |

Each test monkeypatches `promotion.load_campaign_governance` to capture its
kwargs and raise a sentinel exception, so the assertion is made before any
real training, RL, or ladder work would run (consistent with the existing
`test_training_promotion_requires_governance_before_data_load` pattern in
`tests/test_harnesses/experiments/test_ladder_promotion.py`, which the new
module sits alongside).

```
python -m pytest tests/test_harnesses/experiments/test_locked_manifest_digest_threading.py -q
7 passed in 1.76s
```

Regression check -- existing promotion/campaign suites unaffected:

```
python -m pytest tests/test_harnesses/experiments/test_ladder_promotion.py \
  tests/test_autoresearch/test_experiment_campaign.py \
  tests/test_scripts/test_publish_spectral_disposition.py -q
28 passed in 4.34s
```

(`tests/test_harnesses/experiments/test_capacity_ladder.py::test_capacity_arms_match_on_everything_but_tokenizer`
fails identically on a clean `git stash` of this change -- a pre-existing,
unrelated `grammar_ltr_primary` regression, not caused by this change.)

## Honesty and scope

Fixture/local-CPU wiring evidence, not a ship or promotion claim. No model
quality, promotion, or training result is claimed here. The default gate
behavior of every existing promotion run is unchanged (flag defaults to
`False` everywhere); this only adds a way to opt in to the SLM-306 check from
the three real entrypoints. Flipping the *default* to on (making
`canonical_manifest_path()` implicit rather than opt-in) remains open, exactly
as SLM-306 scoped it, pending a live end-to-end promotion run to validate
against -- still out of reach in this GPU-less sandbox.

## Version stamps

- `harness.model_build.train`: v25 -> v26
- `model.twotower`: v258 -> v259 (watches `ModelBuildConfig` in
  `harnesses/model_build/config.py`)
- `harness.experiments`: v136 -> v137 (now also watches
  `scripts/resume_climb.py`, `scripts/run_scaling_ladder.py`, and the new test
  module)

See `src/slm_training/resources/versions.json` for the full history notes.
