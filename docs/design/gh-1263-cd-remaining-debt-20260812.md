# GH-1263 continuous-delivery follow-up — landed scope and remaining debt

**Issue:** https://github.com/Tyler-R-Kendrick/slm-training/issues/1263  
**Linear:** [SLM-577](https://linear.app/quickdeploy-ai/issue/SLM-577/gh-1263-follow-up-canonicalize-metric-gaming-archetypes-emptiness)  
**Honesty:** fixture/scratch wiring evidence for fresh-checkout bootstrap and
pre-existing test debt cleanup — not ship.

## Landed in GH-1263 (#1679)

1. **Fresh-checkout bootstrap** — `.claude/hooks/session-start.sh` (remote-only
   via `CLAUDE_CODE_REMOTE=true`) idempotently provisions `.venv`, pinned CPU
   torch, repo-root `npm ci`, and `src/apps/openui_bridge/npm ci`. Registered in
   `.claude/settings.json` `SessionStart`. `scripts/setup_dev_env.sh` now also
   installs the OpenUI bridge deps.
2. **`NODE_OPTIONS`** — already on `main` via `bridge_utils.sanitized_node_env()`
   in `agentv.py` (and lang-core / graphql-js bridges). No additional change
   required; issue text predates that landing.
3. **Named-marker test debt (partial)** — canonicalized fixtures in
   `tests/test_evals/test_emptiness_probe.py` and
   `build_fixture_records()` (`canonicalize_example_template_markers`) so
   `GenerationRequest.from_record` / `TwoTowerModel.from_records` stop failing
   `assert_canonical_template_marker_inventory`.
4. **`test_cold_warm_bench` subprocess** — cold-trial probe script inlines phase
   markers so the subprocess no longer depends on an editable install / pytest
   `pythonpath` injection.
5. **`check_changed` blast radius** — `.claude/**` config-only edits no longer
   fall through to the full `tests/` suite fallback.

## Remaining debt (status)

| Item | Status | Notes |
| --- | --- | --- |
| `metric_gaming._archetypes()` named markers | **closed (SLM-577)** | Production archetypes + negative transforms in `metric_gaming.py` / `oracle_scoring_replay.py` now emit opaque `:slot_N` markers. `_archetypes()` asserts `assert_canonical_template_marker_inventory` on every contract. Oracle `build_fixture_records()` still belt-and-suspenders-canonicalizes. |
| `emptiness_probe.minimal_valid_program` flake under concurrent hook shards | **closed (SLM-577)** | Silent `except Exception: continue` replaced with debug tracing per rejected candidate and a warning when no candidate validates. Gates unchanged; I6 untouched. |
| Grammar packed-decode (`grammar.py:326`) | **not reproduced** | `test_v4_levers::test_generate_batch_requests_consumes_harness_slot_contract` passes on current `main` tip. Treat as environment- or revision-specific until reconfirmed. |

## Verification recipe (fresh container)

```bash
uv sync --extra dev
uv pip install -e . --no-deps
scripts/setup_dev_env.sh   # or rely on session-start hook when CLAUDE_CODE_REMOTE=true
pytest tests/test_evals/test_agentv.py tests/test_evals/test_emptiness_probe.py \
  tests/test_evals/test_oracle_scoring_replay.py tests/test_evals/test_metric_gaming.py -q
python -m scripts.verify_merge_ready --fast
```
