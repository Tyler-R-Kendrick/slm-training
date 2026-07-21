# OpenFeature ↔ research harness levers

Status: **active contract (v1).** Owner: `src/slm_training/flags/`.
Linear: [SLM-342](https://linear.app/quickdeploy-ai/issue/SLM-342/openfeature-integration-for-experiment-levers).
Component: `harness.flags` in `src/slm_training/resources/versions.json`.

**Related (product / dashboard OpenFeature):**
[`openfeature-experiments.md`](openfeature-experiments.md) owns
`slm_training.features` (LaunchDarkly / PostHog / in-memory product flags).
This document is only the **research harness** overlay onto `ModelBuildConfig`
(`slm_training.flags`). Do not merge the two registries.

## Why

Research matrices (E*/X*/P*) are offline ablations with matched controls — they
stay as scoreboard evidence. What they *share* with industry progressive-delivery
systems is the **lever**: a typed, named toggle that changes runtime behavior
(`verified_solver_decode`, `honest_slot_contract`, …).

Those levers were only dataclass/CLI fields. OpenFeature is the CNCF standard
evaluation API for that concern. Speaking it lets us:

1. Evaluate levers with a vendor-agnostic client surface.
2. Attach assignment metadata (`variant`, `reason`, `flag_metadata.experiment_id`).
3. Expose OFREP-shaped evaluation for dashboard / external tools.
4. Swap in flagd / LaunchDarkly / PostHog later without rewriting call sites.

OpenFeature's experimentation grouping is still evolving
([spec#370](https://github.com/open-feature/spec/issues/370)); v1 uses flag
metadata (`experiment_id`, `matrix`, `lever`) rather than inventing a parallel
experiment SDK.

## Concepts

| Concept | In this repo |
| --- | --- |
| Flag key | Curated lever name == `ModelBuildConfig` field (e.g. `verified_solver_decode`) |
| Evaluation context | `targeting_key=run_id`; attributes: `experiment_id`, `matrix`, `model_name`, `context_backend` |
| Variant | String form of the resolved value (`"on"`/`"off"`, or the string/number value) |
| Ruleset | In-process map of flag key → typed default used by `InMemoryProvider` |
| Matrix row | Offline ablation evidence; may *seed* a ruleset for a run, but is not a targeting system |
| Scoreboard | Historical JSON under `docs/design/` — unchanged; not live flag state |

## Precedence

```text
CLI / explicit overrides  >  OpenFeature evaluation (ruleset / provider)  >  ModelBuildConfig defaults
```

Unset flags keep dataclass defaults. Default for every registered boolean lever
is **off** / safe, so decode stays byte-identical until a ruleset or override
enables it.

## Scope

**In v1**

- Zero-dependency OpenFeature-compatible client + in-memory provider.
- Lever registry mapped onto `ModelBuildConfig`.
- `apply_experiment_flags` overlay used by config builders / evaluate path.
- Assignment details recorded on the returned overlay (for train summaries).
- `POST /api/flags/ofrep/v1/evaluate` (OFREP-shaped, read-only).
- Design + tests + version stamp.

**Out of v1**

- Rewriting matrix scoreboards as live targeting.
- Remote providers (flagd / LaunchDarkly / PostHog) — optional later via
  `openfeature-sdk` extra behind the same port.
- Changing ship-gate thresholds or default decode behavior.

## Flag registry (v1)

Boolean levers (default `false`):

- `verified_solver_decode`
- `topology_verified_solver`
- `topology_capsule_solver`
- `honest_slot_contract`
- `asap_decode`
- `grammar_constrained` (when present on config)
- `compiler_search_local_nogoods`

String levers:

- `compiler_decode_mode` (default `"off"`)
- `solver_unknown_policy` (default `"keep_and_rank"`)
- `solver_certificate_mode` (default `"summary"`)

Number levers:

- `solver_max_nodes` (default `512`)
- `decode_min_content` (default `0`)

Registry lives in `src/slm_training/flags/levers.py`. Adding a lever requires
a registry entry + a field on `ModelBuildConfig` (or an explicit skip note).

## API

```python
from slm_training.flags import (
    EvaluationContext,
    InMemoryProvider,
    FlagClient,
    apply_experiment_flags,
    experiment_context,
    ruleset_from_mapping,
)

ctx = experiment_context(run_id="e53-…", experiment_id="E53", matrix="quality")
client = FlagClient(InMemoryProvider(ruleset_from_mapping({
    "verified_solver_decode": True,
})))
config, assignments = apply_experiment_flags(config, client=client, context=ctx)
```

OFREP-shaped HTTP:

```http
POST /api/flags/ofrep/v1/evaluate
{"context":{"targetingKey":"run-1","experiment_id":"E53"},"flags":["verified_solver_decode"]}
```

## Honesty

- Fixture / scratch matrix runs that flip flags are still not ship claims.
- Flag evaluation never weakens `--ship-gates`.
- A missing provider or unknown flag returns the typed default with
  `reason=DEFAULT` / `ERROR` — never a silent enable.

## Follow-ups

1. Optional `openfeature` extra wrapping `openfeature-sdk` + OFREP/flagd providers.
2. Dashboard Experiments page: show live assignment details beside scoreboard rows.
3. Tracking hook emitting `experiment.Assigned` / OTEL attrs when OpenFeature
   experimentation conventions stabilize.
