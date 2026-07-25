# SLM-338 / AP-037: unified abstract-planning evidence publisher (slm338_evidence_publisher_fixture)

Status: **published_fixture** · Claim class: **diagnostic** (fixture scale — no promotion or ship claim)

## What shipped

- `AbstractPlanningResultV1` (`src/slm_training/harnesses/autoresearch/planning_result.py`):
  one frozen, content-addressed result object carrying schema id, locked campaign
  manifest sha256 (SLM-337 `campaign_manifest_sha256` digest), code/model/data
  provenance (source commit, dirty flag, checkpoint `ArtifactRef`, data snapshot
  sha), raw/constrained/repaired metric paths (meaning-v2, binder F1, parse),
  plan controls (oracle/random/empty/shuffled), causal interventions,
  latency/compute breakdown (plan/generation/verification/total/p95), verifier
  gates, AgentV + human-audit references, claim class, and `version_stamp`.
- `publication_blockers(result)` — fail-closed: missing/UNKNOWN provenance,
  missing manifest hash, missing total latency, and (for `promotion_candidate` /
  `ship_gate`) missing raw/constrained/repaired paths, negative plan controls,
  or a checkpoint reference each block with a named reason.
- `scripts/publish_planning_result.py` — one command: validates blockers (exit 1
  with reasons), then emits canonical JSON (+ content sha), narrative Markdown
  rendered from the JSON only, a model-card roster row (stdout / `--snippet-out`;
  MODEL_CARD.md is **not** auto-edited), and an idempotent append-only
  disposition section in `docs/design/abstract-planning-evidence-contract.md`
  keyed by campaign hash. Historical iter docs stay immutable.
- Component `harness.experiments.slm338_evidence_publisher` v1 registered in
  `versions.json`.

## Fixture publication run

Command (fixture `tests/fixtures/planning_result/ap037_fixture.json`, diagnostic
claim class, locked manifest sha
`00601c6d…215f41` verified against `tests/fixtures/planning_result/ap037_campaign.json`):

```bash
python -m scripts.publish_planning_result tests/fixtures/planning_result/ap037_fixture.json \
  --output docs/design/abstract-planning-ap-037-fixture \
  --snippet-out outputs/experiments/slm338/model_card_row.md \
  --append-disposition docs/design/abstract-planning-evidence-contract.md
```

Artifacts: [canonical JSON](abstract-planning-ap-037-fixture.json) ·
[narrative Markdown](abstract-planning-ap-037-fixture.md) ·
[disposition contract](abstract-planning-evidence-contract.md) · content sha
`2e683dd3…d8d29c`. Every number in the Markdown is rendered from the JSON;
round-trip rendering is asserted deterministic in tests.

## Tests

- `tests/test_harnesses/autoresearch/test_planning_result.py` — 13 passed
  (round-trip, content-sha determinism, each blocker class, promotion vs
  diagnostic requirements, locked-manifest hash reuse).
- `tests/test_scripts/test_publish_planning_result.py` — 6 passed (end-to-end
  publish, MD-from-JSON numbers, fail-closed exit 1, idempotent disposition,
  `--from-dir` merge, iter-doc immutability).

## Honesty

Fixture-scale diagnostic wiring only. No checkpoint promoted or synced, no
production default changed, MODEL_CARD.md and README untouched (snippet printed,
not inserted).
