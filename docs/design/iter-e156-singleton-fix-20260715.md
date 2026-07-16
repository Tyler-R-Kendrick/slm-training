# E156–E158 constrained decoder follow-up (2026-07-16)

These are diagnostic CPU runs on the source-controlled `remediated_roots_judged`
corpus (`data_manifest_sha=ef09af685063bace02797ef0138cb3e6e238159a66bb99d5c404f57f64505758`),
with local HF context, compositional tokens, no DESIGN.md context, and honest
three-record smoke feedback. All AgentEvals bundles and raw telemetry remain
under the corresponding `outputs/runs/` directories.

| Iteration | Change | Loss | Parse | Structure | Placeholder validity | p50 ms | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| E156 | singleton legal-token return, 8 steps | 27.8651 | 0.0 | 0.2428 | 0.4000 | 13316 | no quality promotion |
| E157 | compiler tree decode on E156 checkpoint | — | 0.0 | 0.2428 | 0.4000 | 6180 | latency-only; fallback exposed |
| E158 | matched 64-step train | 9.6653 | 0.0 | 0.2114 | 0.3167 | 5928 | longer training insufficient |

The singleton fix is correct in isolation, but it is not the dominant failure:
the model still produces malformed lexical sequences and zero parse across all
three runs. E158 lowers training loss without improving generated validity, so
more steps alone are not a justified next lever. The next iteration should
inspect token/grammar alignment and then rerun a matched control; do not weaken
ship gates or treat compiler fallback as model competence.

Evidence: [E156 JSON](iter-e156-singleton-fix-20260715.json),
[E157 JSON](iter-e157-compiler-tree-20260715.json), and
[E158 JSON](iter-e158-64step-singleton-fix-20260715.json).
