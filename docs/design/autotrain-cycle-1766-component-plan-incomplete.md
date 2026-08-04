# Autotrain c1766: component-plan measurement is incomplete

**Verdict:** infrastructure-inconclusive; replay the exact frozen arm. Both CPU
scratch arms trained on `wf_smoke_v2` with seed 101766, batch size 2, 20
requested steps, and exactly 1,755,764 trainable parameters. The candidate
enabled component-plan train and decode weights of 1.0; the matched control
used 0.0. Training completed and both checkpoints were written, but the
candidate timed out on all three smoke documents.

| Arm | Params | Train wall | Last weighted loss | Smoke | AgentV / gates | Decision |
| --- | ---: | ---: | ---: | --- | --- | --- |
| component-plan | 1,755,764 | 7.590 s | 21.3778 | n=3; completed 0; timeouts 3; p50 including incomplete 7,538.38 ms; quality unavailable | bundle complete; gate rejection is not expected/scoreable | infrastructure-incomplete |
| matched control | 1,755,764 | 3.229 s | 17.6568 | n=3; completed 3; parse 1; meaning .3333; structure .13527; binder F1 .6333; p50 1,135.88 ms | bundle complete; no execution errors; fail | complete gate rejection |

The strict completeness repair is working as intended: null candidate quality
fields and three runtime timeouts are classified as measurement-integrity and
runtime failures, not as a model-quality result. The supervisor therefore
withholds the primary comparison and emits a content-bound repair/replay action
for frozen candidate manifest
`bf5e918c828e08de6a2f56aa8c574d13892eb947311e8c5919ff299ef53236f5`.

The partial telemetry identifies the runtime shape without licensing a quality
claim. Relative to control, the candidate used 173 versus 11 model forwards,
109,682 versus 31,115 unique completion states, 110,398 versus 31,277 transition
misses, 5,024 versus 861 witness expansions, and 114,270 versus 32,123 parser
forks. The plan bias applied 11 times and changed three choices. Each candidate
document received only about 7.54 seconds from the bounded evaluation wall and
ended at that deadline. This suggests plan-induced search expansion, but the
partial counters do not prove whether a frozen no-retrain replay can finish.

The next cycle must reuse the exact completed train stages and replay both
content-bound manifests before testing another hypothesis. It must not widen
production decode deadlines, weaken constrained decoding, or interpret partial
quality fields. If the bounded replay remains incomplete, component-plan decode
weight 1 is an exhausted runtime-dangerous arm and the canonical owner should
add a measured search-cost guard or a cheaper verified scheduling strategy.

Both checkpoints are local, explicit no-sync diagnostics. The control is
provenance-only and the candidate is restricted to exact frozen replay. Neither
is reusable, promoted, or ship evidence. Lean is
`not_applicable:no_champion`; promotion formal proofs remain mandatory when a
champion exists.

Machine evidence:
[`autotrain-cycle-1766-component-plan-incomplete.json`](autotrain-cycle-1766-component-plan-incomplete.json).
