# Revmath examples (current schemas only)

Hermetic fixtures under `../fixtures/` are the runnable examples. All task
JSON uses `schema_version: "revmath_task/v1"`. Do not advertise or write
retired schemas (`revmath_task/v0`, parallel `RevMathCampaign`, or a profile-
local evidence store).

## Forward theorem (hermetic)

```bash
PYTHONPATH=src uv run python -m scripts.run_revmath_task \
  --task src/slm_training/resources/revmath/fixtures/hermetic_forward_theorem.task.json \
  --hermetic --output-dir outputs/revmath/hermetic
```

## Assumption ablation via locked profile

```bash
PYTHONPATH=src uv run python -m scripts.run_revmath_profile \
  --task src/slm_training/resources/revmath/fixtures/ablation_necessary.task.json \
  --materialize-fixture \
  --campaign-id camp.rm.hermetic \
  --experiment-id exp.rm.hermetic.profile \
  --store-root outputs/revmath/profile_runs \
  --hermetic
```

## Labeling claims (HARN-08)

Positive/negative claims: `../fixtures/labeling_*.claim.json` via
`harnesses/reasoning/revmath/labeling.py`. Genuine Big-Five labels need the
interpretation package; practical computability stays
`practical_computability_only`.

## External solvers

Lean / multi-prover backends are optional experiments (`--hermetic` is the
default). Missing tools map to unknown, never refutation. See
`docs/design/reverse-mathematics-computability.md` § Optional external tools.
