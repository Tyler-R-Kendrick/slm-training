# Training-data pipeline integration

SLM-16 makes `build_train_data --source all` the sole deterministic writer for
the merged P1-P11 data producers. This is corpus-build evidence, not a model
quality, checkpoint, or ship-readiness claim.

## Measured result (2026-07-14)

The fixture-backed CPU reproduction used the official OpenUI bridge, one typed
ProgramSpec root, one RICO screen, every committed integration source, and the
bounded repair/edit/frontier/design-contract derivatives. No training steps or
checkpoint were produced.

| Measure | Result |
| --- | ---: |
| Input seeds | 103 |
| Candidates after synthesis | 210 |
| Kept, revalidated records | 186 |
| F2/governance rejects | 24 |
| Quality rejects | 0 |
| Build errors | 0 |
| First content fingerprint | `2f14233beeca874af171990935ea168cd4c4024fce9c803ce873ee48fc77d9a7` |
| Second content fingerprint | `2f14233beeca874af171990935ea168cd4c4024fce9c803ce873ee48fc77d9a7` |

The accepted corpus contains `programspec_generated`, `language_contract`,
`corruption_repair`, `edit_trajectory`, `frontier_described`,
`abstraction_ladder`, `renderer_visual`, and `web_distilled`, alongside the
eligible legacy fixture/RICO families. Diffusion remains train-time only: the
manifest records all 11 P10 policies and does not write noisy targets. P11's
weights are consumed unchanged and produce mixture hash
`5b3c27c878f71af3d9734271d87c23a08bef2fc8165f5fec73dd7f55420c8397`.

Every kept row carries a fresh F2 tier and no `Quarantine` row is written.
Governance output includes Croissant, Data Card, and SPDX documents. The 24
rejected candidates are durable evidence that governance and grounding gates
remain fail-closed; rejection is not converted into synthetic success.

## Verification recipe

```bash
PYTHONPATH=src \
OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/tools/openui_bridge/cli.mjs \
/home/codex/repos/slm-training/.venv/bin/pytest -q \
  tests/test_data tests/test_dsl tests/test_harnesses/train_data
```

Result: **239 passed, 2 skipped**. The paired build reproduction and complete
recipe are recorded in
[`train-data-pipeline-integration.json`](train-data-pipeline-integration.json).
No ship gates were run because this change does not evaluate a model.
