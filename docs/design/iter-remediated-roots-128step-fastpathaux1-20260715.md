# Structural force-align auxiliary diagnostic — 2026-07-15

The existing `fastpath_aux_weight` objective was enabled at weight 1.0 on the
source-controlled `remediated_roots` corpus, with the matched 128-step
TwoTower scratch diffusion recipe.

- 108 records / 94 unique targets
- 60,846 target tokens; telemetry total 32.82 seconds
- Best held-out weighted NLL: 4.907197
- Broad mean NLL: 5.427811

Corrected constrained smoke (`n=3`, LTR repair, 64-token cap, one attempt)
remained 0.0 parse, 0.0 structural similarity, and 0.0 reward. There were no
decode timeouts; p50 latency was 2,066 ms. Reject the checkpoint. The next
harness audit must verify that this auxiliary loss is active and represented in
telemetry before tuning its weight further.
