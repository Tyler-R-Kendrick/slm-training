# SLM-578: harvest continuous-openui measured-results from agent/autotrain-continuous-live

**Date:** 2026-08-12
**Issue:** [SLM-578](https://linear.app/quickdeploy-ai/issue/SLM-578/harvest-missing-continuous-openui-measured-results-from-agentautotrain)
**Honesty:** docs/harness continuity harvest — not a ship claim. I6 / ship gates untouched.

## Source

| Ref | SHA |
| --- | --- |
| `origin/main` (base) | `ae2f5c8e85eeb3e9c08be9bdaf5d4d0dc1a41333` |
| `origin/agent/autotrain-continuous-live` (tip) | `af90e3eeccf489f7e3082e6fbdfc8a552599d7af` |

Do **not** blind-merge the ~1409-commit tip. Main tip `run_autotrain_continuous.py` is newer (~13.1k lines vs ~10.5k on the live tip).

## Docs landed (97 paths absent on main)

All paths were present only under `docs/design/` on the live tip (zero non-doc residuals vs main tree).

| Family | Count |
| --- | ---: |
| `continuous-openui-local-*` | 92 |
| `autotrain-cycle-*` | 2 |
| `decode-timeout-repair-self-heal*` | 2 |
| `dual-arm-decode-timeout-escape-*` | 1 |
| **Total** | **97** |

## Code audit (no port)

| Candidate | Verdict |
| --- | --- |
| `f37eb055d` / `8530bc0af` / `f8090f815` decode-timeout `repair_harness` self-heal | **Already superseded on main** via `_self_heal_thrash_timeout_repair` + thrash timeout residual routing in `scripts/run_autotrain_continuous.py` (and tests). Live tip would regress the driver (~3k lines deleted vs main). |
| `thrash_regime.py` / climb `policy.v1.json` / engine+schemas | **Already on main**; live `policy.v1.json` is behind main (would drop newer policy). |

## Residual

None under the live tip tree vs main after this harvest. Safe to delete remote `agent/autotrain-continuous-live` after land.

## Follow-ups (out of scope)

- I13 encoder ops-conditioning campaign
- External-judge / blinded-human successors (SLM-483 etc.)

## version_stamp backfill

Two newly added result-shaped JSON files carried a top-level `gates` key and
therefore failed `verify_version_stamps` (new `docs/design/*.json` must stamp).
Stamped with `build_version_stamp(*EVAL_KEY_COMPONENTS)`, `code_commit` set to
each file's historical `integration_commit`, `stamped_at` = `recorded_at`.
Other harvested continuous cycle JSON lacked RESULT_SHAPE_KEYS and did not
require stamps.
