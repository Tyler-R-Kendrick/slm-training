# Iteration: batch-8 seed stability check (2026-07-15)

Two four-step scratch TwoTower runs used the same remediated corpus,
`batch_size=8`, `lr=6e-4`, final-only loss feedback, and identical model/data
configuration. Seed 0 reached weighted held-out NLL **27.977** at 6,252 target
tokens; seed 1 reached **31.165** at 5,518 target tokens. Both persisted
complete final loss reports, run insights, and telemetry.

The top-five per-example token-loss IDs had **zero overlap** between seeds.
Seed 0's highest example was `rico_train_96_syn_0`; seed 1's was
`rico_train_13_syn_2`. Because the tail is seed-sensitive, no example deletion,
reweighting, or corpus rebalance is justified. Keep the proxy as diagnostic
telemetry and require a larger multi-seed sample before data intervention.

This is a scratch diagnostic and does not establish a ship claim.
