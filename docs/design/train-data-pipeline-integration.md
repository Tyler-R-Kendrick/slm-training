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
| Kept, revalidated records | 184 |
| F2/governance rejects | 26 |
| Quality rejects | 0 |
| Build errors | 0 |
| First content fingerprint | `805cafdae01bc8e696b8d3a833dcc794f90dff98e2032dcd1d317db84b8992fc` |
| Second content fingerprint | `805cafdae01bc8e696b8d3a833dcc794f90dff98e2032dcd1d317db84b8992fc` |

The accepted corpus contains `programspec_generated`, `language_contract`,
`corruption_repair`, `edit_trajectory`, `frontier_described`,
`abstraction_ladder`, `renderer_visual`, and `web_distilled`, alongside the
eligible legacy fixture/RICO families. Diffusion remains train-time only: the
manifest records all 11 P10 policies and does not write noisy targets. P11's
weights are consumed unchanged and produce mixture hash
`509044358bcf846884d5154f28e1b9d492c79fcef43ebcd68c0406c1acad600c`.

Every kept row carries a fresh F2 tier and no `Quarantine` row is written.
Governance output includes Croissant, Data Card, and SPDX documents. The 26
rejected candidates are durable evidence that governance and grounding gates
remain fail-closed; rejection is not converted into synthetic success.

## Verification recipe

```bash
PYTHONPATH=src \
OPENUI_BRIDGE_CLI=/home/codex/repos/slm-training/tools/openui_bridge/cli.mjs \
/home/codex/repos/slm-training/.venv/bin/pytest -q \
  tests/test_data tests/test_dsl tests/test_harnesses/train_data
```

Result: **237 passed, 2 skipped**. The paired build reproduction and complete
recipe are recorded in
[`train-data-pipeline-integration.json`](train-data-pipeline-integration.json).
No ship gates were run because this change does not evaluate a model.
