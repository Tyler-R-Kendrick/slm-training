# E1020-E1022 — fast binder-component training

Date: 2026-07-22. CPU scratch work under the repository wall cap.

Model v267 replaces binder-component training's full compiler-forest replay
with a linear scan over grammar-native declaration tokens. Focused Form/Tabs
tests prove identical `(binder, component)` labels, while the existing
gradient/decode test remains green. The change adds no target text, literals,
fixtures, candidate filtering, or component-specific cases.

E1020 completes 450 fresh E937 steps in 61.04 seconds with batch size 4,
compared with E1015 stopping after 59 steps in 96.65 seconds. Measured example
throughput rises from 2.44/s to 29.49/s (12.08x). The run sees 1,800 examples,
uses no parent, and explicitly disables checkpoint sync. Checkpoint SHA is
`d0d792e2...826225`.

| Run | Suite | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1021 | smoke | 3 | 1.0 | 1.0 | 1.0 | 0.5453 | 0.5 | 0.9490 | 0 / 0 |
| E1022 | held_out | 5 | 0.6 | 0.4 | 0.4333 | 0.3050 | 0.2952 | 0.5062 | 2 / 3 |
| E996 baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

Both evals emit AgentEvals JSONL and pinned AgentV bundles (`0/1` each).
Smoke is clean, but held Form still collapses, Tabs emits an empty `Tabs` plus
inline buttons, and two rows time out. The binder head records zero
applications on both suites; its final sampled-batch accuracy is also zero.
The smoke gain therefore does not demonstrate useful binder-plan decoding.

Retain the v267 label-extraction optimization. Reject E1020 and never sync,
promote, serve, resume, or use it as a parent. A follow-up should detach or
downweight the auxiliary gradient so an uncalibrated head cannot reshape shared
prompt features before it learns a useful ranking.
