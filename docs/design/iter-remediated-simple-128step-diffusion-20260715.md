# Simple-target corpus diagnostic — 2026-07-15

Built a filtered candidate from the persisted `remediated_roots` records using
`max_openui_chars=180` and `max_components=4`, with no synthesis or children.
The builder emitted 137 records after quality filtering; candidate fingerprint
is `1c30c46f1e98b61b90edb1e5a60883e6d354e3ee825518b04f34622b5bdf086c`.

The 128-step TwoTower scratch diffusion run used batch size 8, LTR loss weight
2.0, and CPU telemetry (49,120 target tokens; 31.91 seconds total). Best
held-out weighted NLL was 4.962872 and broad mean NLL was 5.752690.

Corrected constrained smoke (`n=3`, one attempt, LTR repair, 64-token cap)
remained 0.0 parse, 0.0 structural similarity, and 0.0 reward. There were no
decode timeouts; p50 latency was 2,311 ms. The candidate is rejected. Target
length/component filtering alone does not explain the generation collapse.
