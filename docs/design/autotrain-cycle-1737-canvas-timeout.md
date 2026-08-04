# Autotrain c1737: compact-canvas incomplete measurement

**Verdict:** no control-versus-candidate result exists. Both exactly matched
22-step trains completed, but both smoke evaluations reached the wall cap after
two of three records while decoding `smoke_callout_01`. Their progress
artifacts are explicitly non-scoreable. Quality, latency, gates, and AgentV
results are unavailable; the campaign is inconclusive and requires an exact
frozen replay.

## Result matrix

| Arm | Params | Train | Loss | Eval progress | Forwards | Completion states | Metrics / gates | Disposition |
| --- | ---: | --- | ---: | --- | ---: | ---: | --- | --- |
| matched control | 1,608,962 | 22 steps / 3.127s | 9.5914 | 2/3 processed; interrupted on `smoke_callout_01` | 88 | 276,746 | unavailable / not run | incomplete |
| compact active canvas | 1,608,962 | 22 steps / 3.025s | 9.5914 | 2/3 processed; interrupted on `smoke_callout_01` | 88 | 275,313 | unavailable / not run | incomplete |

The partial decode counters are telemetry only. They may not be used to claim a
runtime or quality delta because neither arm finalized the suite. No AgentV
bundle or gate result was emitted. The checkpoints are local, explicit no-sync,
and may be reused only by the exact frozen-replay mechanism.

## Harness signal and repair

Phase A correctly wrote `measurement_complete=false`, but two downstream
surfaces lost that fact. The cycle handoff recognized only a narrow
`measurement_incomplete:*` reason, while the outcome diagnosis treated the
training-only `trainable_params` metric as evidence that a stopped experiment
had completed. That yielded a false model diagnosis and proposed the next
`both` lever instead of replaying canvas.

Harness v57 now treats the explicit completeness field as authoritative,
classifies stopped outcomes with partial training metrics as infrastructure,
and makes the exact frozen retry the leading terminal priority. The regenerated
handoff is `inconclusive` and queues candidate manifest
`763ae28e…d96d` at retry count 0/2. A new model hypothesis cannot run before
that typed retry is consumed.

Lean is `not_applicable:screening` because no completed candidate or promotion
decision exists; the full repository Lean proof suite was green before this
run. Machine-readable evidence is in
[`autotrain-cycle-1737-canvas-timeout.json`](autotrain-cycle-1737-canvas-timeout.json).
