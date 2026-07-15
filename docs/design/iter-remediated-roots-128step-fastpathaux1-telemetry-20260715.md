# Verified structural auxiliary run — 2026-07-15

After adding explicit telemetry and removing silent exception suppression, the
fastpath force-align auxiliary objective was rerun on `remediated_roots`.

- TwoTower scratch / CPU, diffusion mask, 128 steps, batch 8
- 108 records / 94 unique targets; 60,846 target tokens
- `fastpath_aux_weight=1.0`, `ltr_loss_weight=2.0`
- Best held-out weighted NLL: 4.907197
- Auxiliary span: 128 calls, 151.806 ms total, 1.186 ms mean

Constrained smoke (`n=3`, LTR repair, 64-token cap, one attempt) remained 0.0
parse, 0.0 structural similarity, and 0.0 reward. There were no decode
timeouts; p50 latency was 2,097 ms. Reject the checkpoint. The auxiliary loss
is active and measured, but does not solve structural generation.
