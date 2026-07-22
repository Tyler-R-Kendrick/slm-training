# E734 root-arity rank calibration

**Date:** 2026-07-22  
**Decision:** reject and revert rank-calibrated arity decoding  
**Evidence:** [`iter-e734-root-arity-rank-calibration-20260722.json`](iter-e734-root-arity-rank-calibration-20260722.json)

E734 tested whether lexer root-arity weight 1 should interpolate all the way to
the trained continue-or-stop class when that class conflicts with compiler path
scores. The implementation was committed before evaluation, and unit tests
proved it could reverse both stop and continue conflicts in the canonical tree
and restricted compiler paths.

The clean-revision local CPU smoke result is decisively negative. The weight-0
control reproduces E731/E723: parse 1.0, meaning-v1 0.6667, fidelity 0.5278,
structure 0.5614, recall 0.4167, reward 0.8073, and no timeouts. Weight 1 times
out all three records at the per-record eight-second limit and consequently
scores zero on every quality metric. Both arms publish AgentV 0/1 bundles.

The treatment is therefore rejected and reverted; `model.twotower` v197
restores v195's bounded raw-logit integration. No checkpoint was created,
synced, promoted, or changed. This closes rank-forcing on the weak E731 arity
head. The next arm must improve the structural target or its eligible training
distribution before granting that head stronger decode authority.
