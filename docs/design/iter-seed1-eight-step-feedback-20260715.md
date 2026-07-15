# Iteration: seed-1 eight-step loss versus generation feedback (2026-07-15)

Seed 1 used the same scratch TwoTower recipe as the seed-0 eight-step control:
batch size `8`, learning rate `6e-4`, eight steps, compositional tokenizer,
and final-only loss feedback. The run consumed **12,258** target tokens and
persisted training telemetry, a checkpoint, and the deterministic loss-suite
artifact. Weighted held-out NLL was **31.841**.

The comparable seed-0 run reached **17.410** at 11,122 target tokens. This
large two-seed spread is evidence of short-run variance, not evidence that the
corpus or recipe should be reweighted from this observation alone.

A bounded unconstrained smoke evaluation of the seed-1 checkpoint completed
and persisted AgentV artifacts, but parse rate, structural similarity, and
reward were all **0**. The constrained path remains too slow to score. The
checkpoint is not promoted and neither loss improvement nor loss regression is
treated as generation readiness.

Recipe metadata: CPU, scratch context, unfrozen context tower, effective batch
size 8, final-only loss evaluation, diagnostic smoke evaluation with one record,
one generation step, and one attempt. This is a scratch diagnostic, not a ship
claim.
