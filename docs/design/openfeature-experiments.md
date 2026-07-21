# OpenFeature representation of experiments

Status: implemented (wiring + unit evidence; no train/eval run involved).
Component: `harness.autoresearch.openfeature` (v1) in
`src/slm_training/resources/versions.json`.

## Why

Experiment assignment in this repo is explicit and typed — an
`ExperimentSpec` changes allowlisted `ExperimentKnobs`, and
`compile_commands` turns those knobs into bounded CLI argv
(`src/slm_training/autoresearch/`). That contract is sound but proprietary:
nothing outside this codebase can ask "which configuration does experiment X
run under?" without parsing our Pydantic artifacts.

[OpenFeature](https://openfeature.dev) is the CNCF standard for feature
flagging, and experiments map onto it cleanly:

| Autoresearch concept | OpenFeature concept |
| --- | --- |
| Allowlisted knob (`ExperimentKnobs` field) | Flag key |
| Experiment (`ExperimentSpec`) setting a knob | Variant (named by `experiment_id`) |
| Which experiment a run belongs to | Evaluation context (`experiment_id` attribute / targeting key) |
| Unset knob (`None`) → harness default | Code-defined default (`defaultVariant: null`, reason `DEFAULT`) |
| Knob applied by the targeted experiment | Reason `TARGETING_MATCH` |
| Non-allowlisted knob | `FLAG_NOT_FOUND` error |

## Design

Owner: `src/slm_training/autoresearch/openfeature.py` (extends the existing
autoresearch owner; no parallel experiment store). Two surfaces:

1. **flagd export** — `export_flagd_flags(experiments, flag_set_id=...)`
   renders a set of experiments (typically one `HypothesisMatrix`) as a flagd
   flag definition (`$schema: https://flagd.dev/schema/v0/flags.json`). One
   flag per knob any experiment sets; variants keyed by experiment id;
   JsonLogic targeting resolves the `experiment_id` context attribute to its
   variant and returns `null` (→ code default) otherwise. Zero new runtime
   dependencies; the artifact is consumable by flagd and every OpenFeature SDK.
2. **In-process provider** — `ExperimentFlagProvider` implements the
   OpenFeature `AbstractProvider` over the same specs, so Python consumers
   evaluate knobs through the standard client API
   (`client.get_integer_details("steps", 200, ctx)`). Requires the optional
   `openfeature-sdk` dependency: `pip install slm-training[openfeature]`
   (also included in `[dev]` so tests cover it; default installs stay
   hermetic — the module imports fine without the SDK and only the provider
   class raises).

Fail-closed semantics (mirrors the typed-knob contract):

- Flag key not an `ExperimentKnobs` field → `FLAG_NOT_FOUND`.
- No `experiment_id` attribute or targeting key → `TARGETING_KEY_MISSING`.
- `experiment_id` not among the provider's experiments → `INVALID_CONTEXT`.
- Requested type does not match the knob's typed value → `TYPE_MISMATCH`
  (bool is never an integer/float; ints widen to float only for float
  resolution).
- Knob unset on the targeted experiment → caller's default, reason `DEFAULT`.

## CLI

```bash
python -m scripts.autoresearch export-openfeature \
  --campaign-id <id> [--matrix path/to/matrix.json] [--output flags.flagd.json]
```

Defaults to the campaign's latest formed hypothesis matrix, writes the
document as a content-addressed `openfeature` artifact in the
`CampaignStore`, appends an `openfeature_exported` event, and prints the
flags JSON.

## Scope and non-goals

- **Selection stays explicit.** OpenFeature here *represents* experiment
  configuration; it does not introduce percentage rollouts or runtime
  reassignment. The hypothesizer's `recommended_experiment_id`, matrix
  `--only` filters, and campaign budgets remain the only assignment
  mechanisms.
- **Gates are not flags.** Ship gates, promotion criteria, and honesty
  policy (`harness_core` gate/promotion engines) are measurement policy and
  stay outside OpenFeature entirely.
- **One experiment world first.** Hand-authored matrix rows
  (`run_quality_matrix.Experiment`, `PerfExperiment`) are not exported; they
  can adopt `export_flagd_flags` later if a consumer appears (their rows are
  frozen dataclasses, so the mapping is mechanical).
- No flagd daemon, no remote flag source, no OpenFeature hooks/events —
  YAGNI until an external consumer exists.

## Evidence

`tests/test_autoresearch/test_openfeature.py` (9 tests): flagd document
shape and targeting rule, duplicate/empty rejection, matrix export, CLI
export round-trip (store artifact + event + `--output` file), and — through
the real OpenFeature client — targeting match for all five flag types,
code-default fallback, targeting-key selection, and every fail-closed error
code. `pytest tests/test_autoresearch/test_openfeature.py` → 9 passed.
The repo-wide suite shows no regression: 183 pre-existing failures + 5
errors are identical on `main` and this branch (verified side by side).
