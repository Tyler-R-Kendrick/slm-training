# GH-1263 continuous-delivery follow-up — landed scope and remaining debt

**Issue:** https://github.com/Tyler-R-Kendrick/slm-training/issues/1263  
**Honesty:** fixture/scratch wiring evidence for fresh-checkout bootstrap and
pre-existing test debt cleanup — not ship.

## Landed in this PR

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

## Remaining debt (honest, pre-existing on main)

| Item | Status | Notes |
| --- | --- | --- |
| `metric_gaming._archetypes()` named markers | **open** | Production adversarial archetypes still use semantic names (`:card.title`, …). `metric_gaming` tests pass because they do not call `assert_canonical_template_markers`. Oracle replay now canonicalizes at fixture build time only. Full archetype canonicalization needs a careful audit of every negative transform that references named slots. |
| `emptiness_probe.minimal_valid_program` flake under concurrent hook shards | **open** | Issue §3c: `except Exception: continue` can swallow transient grammar/backend errors under load; passes standalone. Needs tracing/logging, not gate weakening. |
| Grammar packed-decode (`grammar.py:326`) | **not reproduced** | `test_v4_levers::test_generate_batch_requests_consumes_harness_slot_contract` passes on current `main` tip (`4a05eb8e1`). Treat as environment- or revision-specific until reconfirmed. |

## Verification recipe (fresh container)

```bash
uv sync --extra dev
uv pip install -e . --no-deps
scripts/setup_dev_env.sh   # or rely on session-start hook when CLAUDE_CODE_REMOTE=true
pytest tests/test_evals/test_agentv.py tests/test_evals/test_emptiness_probe.py \
  tests/test_evals/test_oracle_scoring_replay.py -q
python -m scripts.verify_merge_ready --fast
```
