# Autotrain cycles 1708–1710: canonical stage-artifact recovery

**Evidence class:** local CPU fixture/scratch. **Disposition:** infrastructure
failure; no model-quality, promotion, or ship claim. Machine-readable evidence:
[`autotrain-cycles-1708-1710-runtime-artifact-recovery.json`](autotrain-cycles-1708-1710-runtime-artifact-recovery.json).

## Result matrix

All six `wf_smoke_v2` training arms completed their declared steps and wrote local,
explicit no-sync checkpoints. Evaluation never started: the campaign process helper
correctly retained only an 8 KiB log tail, but the engine then tried to parse that
tail as the typed training result instead of reading the complete canonical
`train_summary.json`.

| Cycle | Arm | Steps | Batch | Params | Last loss | Train wall | Eval / AgentV | Gates | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 1708 | both | 81 | 2 | 1,608,962 | 4.0570 | 7.02s | not run | unavailable | infrastructure / inconclusive |
| 1708 | control | 81 | 2 | 1,608,962 | 4.0570 | 10.84s | not run | unavailable | infrastructure / inconclusive |
| 1709 | control | 82 | 2 | 1,608,962 | 12.3192 | 8.98s | not run | unavailable | infrastructure / inconclusive |
| 1709 | steps | 164 | 2 | 1,608,962 | 3.2810 | 17.43s | not run | unavailable | infrastructure / inconclusive |
| 1710 | batch1 | 80 | 1 | 1,608,962 | 4.1674 | 6.74s | not run | unavailable | infrastructure / inconclusive |
| 1710 | control | 80 | 2 | 1,608,962 | 12.1197 | 9.50s | not run | unavailable | infrastructure / inconclusive |

Loss is training telemetry, not a quality verdict. It is not comparable across the
changed step/batch recipes as evidence of a model improvement, and no row has a
held-out suite, AgentV bundle, or ship-gate result.

## Signals and correction

| Signal | Authority | What it means | Action |
| --- | --- | --- | --- |
| Declared steps, `stopped_on=steps`, and checkpoints are present | observed canonical train summaries | Training itself completed | Preserve the checkpoints as provenance; do not promote or reuse them |
| Outcome says `training produced no typed summary` | contradicted by canonical artifacts | Harness classification is false | Resolve typed output from the current stage's canonical artifact |
| No `eval.json` or AgentV bundle exists | observed | Model quality is unknown, not failed | Replay the identical frozen c1710 arms after the repair |
| Six arms use the same 1,608,962 parameters | observed | Capacity is matched | Keep size fixed during replay; do not buy a green result with growth |

The repair keeps bounded stdout/stderr tails as logs, but treats complete stdout or
a canonical `train_summary.json` / `eval.json` refreshed by the current stage as
the typed authority. Unchanged artifacts from an earlier attempt remain fail-closed.

## Next-run priority

The first priority is measurement integrity, not a new model hypothesis: replay the
locked c1710 batch-size arm and control under manifest
`a43dee3a3b06fcbdaaf33f26e74bac1100f6329fd897a7a53bb7258d394f2ba3`.
Only a complete held-out scoreboard plus AgentV may attribute a quality effect.
After that replay, the loop may resume its ranked `bounds` hypothesis; Lean/formal
preflight remains a required promotion prerequisite and is not bypassed by this fix.
