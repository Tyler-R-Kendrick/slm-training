# Iteration: LTR-weight-2 longer run with checkpoint selection (2026-07-15)

The structure-oriented `ltr_loss_weight=2` recipe was extended to 128 steps
with interval loss feedback every 32 steps. It used the unchanged seed-0,
batch-8 scratch recipe and the same 585-record corpus.

Held-out weighted NLL was 8.955 at step 32, 7.294 at step 64, 8.123 at step
96, and 7.575 at step 128. The harness selected `best_weighted_nll.pt` at
step 64. Bounded constrained evaluation of that checkpoint reproduced
structural similarity **0.5375**, component recall **0.5**, placeholder validity
**0.2**, parse **0**, and reward **0**. The final 128-step checkpoint was not
promoted.

Decision: retain the weight-2/step-64 checkpoint as the best structure-oriented
candidate, but do not call it generation-ready. The next intervention must
close the remaining syntax/serialization gap rather than increase training
steps blindly. All interval telemetry, checkpoints, scoreboards, and AgentV
artifacts were persisted. Scratch diagnostic only.
