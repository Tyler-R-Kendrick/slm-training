# SLM-196: canonical legal-edit bridge fixture

**Disposition:** fixture pass; production publication and downstream model
work remain blocked.

**Machine-readable result:**
[`iter-slm196-legal-edit-bridge-20260723.json`](iter-slm196-legal-edit-bridge-20260723.json)

## Result

The CPU development fixture built 4 non-terminal rows across 2 target clusters
and 2 two-edit bridges. All row transitions replayed, all four exact
candidate-set artifacts reconstructed, split/cluster checks passed, no
forbidden model-input fields were observed, and no records were rejected.
Every row was multi-positive. Candidate sets ranged from 9 to 19 edits
(mean 13.5); 24.07% of candidates remained `UNKNOWN`, and none were treated as
negative.

| Gate | Result |
| --- | ---: |
| source/transition replay | 4/4 (100%) |
| exact candidate reconstruction | 4/4 (100%) |
| fixture target reachability | 2/2 (100%) |
| multi-positive rows | 4/4 (100%) |
| split/target-cluster safety | pass |
| confirmation rows | 0 |
| rejected records | 0 |
| production publishable | **no** |

The fixture exercised `InsertStatement` and `InsertChild`. Its candidate-size
histogram was `{9: 1, 10: 1, 16: 1, 19: 1}`; the singleton ratio was zero.
Target-cluster ICC inputs and per-row counts are retained in the fixture
manifest.

## Honest decision

The shared ragged batch and content-addressed corpus boundary work on the
fixture, including stable semantic IDs, request-local pointer features,
permutation-canonical packing, singleton support tests, multi-positive targets,
and a distinct UNKNOWN mask.

This run does **not** establish the hypothesis’s coverage gain. The frozen
SLM-189 result is fixture-only/inconclusive rather than a selected production
planner, and no hash-pinned X22 or solver-trace baseline manifest was supplied.
The builder therefore reports both coverage deltas as unavailable and rejects
the development planner manifest in production mode. Downstream training must
remain blocked until those inputs satisfy the production gate.

## Recipe and command

- Device: CPU
- Steps/checkpoint: none
- Backend: canonical statement edit algebra
- Matrix set: `slm196_legal_edit_bridge`
- Suite n: 2 records, 4 rows
- Honesty mode: development fixture, confirmation disabled
- Hard wall cap: 3 minutes; observed build completed in under 1 second

```bash
PYTHONPATH=src .venv/bin/python -m scripts.build_legal_edit_bridges --fixture
```

The required `quality_report.json`, empty `rejected.jsonl`, and
`synthesis_feedback.json` were read after the build. Feedback status was
`pass` with no producer repair recommendation.
