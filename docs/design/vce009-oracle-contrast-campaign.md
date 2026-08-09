# VCE-009 (SLM-468): oracle/contrast fixture campaign

New module: [`src/slm_training/harnesses/experiments/vce009_oracle_contrast_campaign.py`](../../src/slm_training/harnesses/experiments/vce009_oracle_contrast_campaign.py).
Runner: [`scripts/run_vce009_oracle_contrast_campaign.py`](../../scripts/run_vce009_oracle_contrast_campaign.py).
Follows the [PCT-008](certified-completion-artifact-and-tps-target.md) pattern
(`harnesses/experiments/pct008_artifact_cold_warm.py`): a frozen dataclass with a
`.manifest()` method returning an `ExperimentCampaignV1`, and a `run_campaign()`
function that locks the manifest via `CampaignStore`, runs real arms, and persists a
plain-dict result via `store.write_artifact`/`append_event` -- per the ownership
map's pre-registered `experiment_campaign` extension point for this issue
(`new_owner_justified: false`, "`ExperimentCampaignV1` is the existing
governed-evidence-envelope owner").

## What this runs

Ten arms against one frozen fixture slice (six deterministically generated
`SemanticPlanV1`s, seed 3):

**Oracle arms** (VCE-005, `data/semantic_plan/oracle.py`), all through the real
`apply_plan_intervention`:
- `oracle_baseline` (control) -- `build_baseline_intervention`
- `oracle_one_factor_{archetype,roles,topology,bindings}` -- one-factor `predicted`-source substitution per factor
- `oracle_all_factors` -- all four factors declared at once
- `oracle_shuffled` -- `select_shuffled_oracle` against the rest of the frozen slice
- `oracle_destructive` -- `plan_source="destructive"` permutation

**Contrast/metamorphic arms** (VCE-006/VCE-007, `data/semantic_contrast/`):
- `contrast_corpus_scoreboard` -- a real, small `SemanticContrastBuilder` build (6 wide sources, `strict_delta=True`), reusing its own scoreboard rather than re-deriving pass/fail counts
- `metamorphic_generators` -- runs all five VCE-007 generators once (alpha-rename, sibling reorder, prompt single-fact edit, prompt paraphrase invariance, AST rewrite equivalence) and checks each against its own declared contract

Every arm reports a `disposition` (`"match"` / `"mismatch"` / `"inconclusive"`) and a
`compute` dict (`forwards`/`verifier_calls`/`wall_ms`) with the *same three keys*
across every arm, so the acceptance criterion "all matched arms expose comparable
compute/exposure fields" is checked by construction, not by convention.

## How this satisfies each acceptance criterion

- **"Evidence validates under current campaign/AgentEvals-style schemas"** -- the
  manifest is a real `ExperimentCampaignV1` (not a hand-built dict); `pydantic`
  validation runs on construction. `test_manifest_is_a_real_valid_campaign_with_ten_arms`.
- **"All matched arms expose comparable compute/exposure fields"** -- every arm's
  `compute` dict has the identical three keys; asserted in
  `test_run_campaign_end_to_end_with_real_arms`.
- **"Fixture-only status is machine enforced"** -- `claim_class` is validated to
  `== "fixture"` in `Vce009CampaignV1.__post_init__` (raises otherwise), and the
  promotion path (`harnesses/experiments/promotion.py`'s `claim_class_not_promotable`
  governance check) structurally rejects anything that isn't
  `promotion_candidate`/`ship_gate` -- this campaign can never reach it.
- **"Oracle contamination banner survives all exports"** -- every arm's result row
  carries `contamination_banner`/`is_contaminated` verbatim from its
  `PlanInterventionRecordV1` (never re-derived), and the top-level result also
  aggregates `contaminated_arm_ids`. Since no arm here intentionally uses
  `plan_source="gold"`, none should ever be bannered by construction -- backed by a
  genuine (not vacuous) rollback gate, `unexpected_oracle_contamination`
  (`operator="gt", threshold=0.0`), that fires on any real leak and does not fire at
  the honest value of zero. `test_manifest_rollback_gate_is_not_vacuous`.
- **"Record null/negative/incomplete results as first-class outcomes"** --
  `oracle_shuffled` returns `disposition="inconclusive"` with an explicit
  `reason="no_compatible_candidate_in_frozen_slice"` when no compatible candidate
  exists in the frozen slice, rather than silently reporting a match; forced and
  verified via `test_shuffled_arm_reports_inconclusive_not_fabricated_when_no_compatible_candidate`.

## Evidence

Recipe (deterministic, ~4s per run):

```bash
python -m scripts.run_vce009_oracle_contrast_campaign --mode fixture
```

Full stamped result:
[`vce009-oracle-contrast-results.json`](vce009-oracle-contrast-results.json).

| Metric | Value |
| --- | --- |
| Arms | 10 (1 control, 9 candidates) |
| `arm_contract_match_rate` | 0.9 (9/10 match) |
| `inconclusive_count` | 1 (`oracle_shuffled` -- no compatible candidate in this particular frozen slice; a real, honest outcome, not a bug) |
| `contaminated_arm_ids` | `[]` |

Verified deterministic: two independent runs against separate output roots produce
an identical `manifest_sha256` and identical per-arm dispositions
(`test_run_campaign_is_deterministic_on_repeat`).

Regression coverage: `tests/test_harnesses/experiments/test_vce009_oracle_contrast_campaign.py`,
9/9 passing.

## Honest scope

- This is fixture-scale evidence over six deterministically generated sources, not
  a production-representative sample -- consistent with `claim_class="fixture"` and
  the deliberately vacuous `fixture_only` promotion gate (mirrors PCT-008's own
  scoping).
- `oracle_shuffled` being inconclusive at the default seed/source-count is expected
  and reported honestly rather than tuned away; a larger `--source-count` would
  likely find a compatible pair more often, but that is a sampling-richness
  question, not a correctness one.
- No new persisted immutable dataset was published; this campaign reads the
  existing `SemanticContrastBuilder`/`metamorphic.py` code paths and writes its own
  scratch corpus under a temporary directory per run (never touching
  `openui_hard_valid_v1`).
