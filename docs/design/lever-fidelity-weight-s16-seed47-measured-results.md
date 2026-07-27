# fidelity_loss_weight lever seed47 s16 lr1e-3 bs2 ASAP t30 (NOT SHIP)
baseline fidw=0.5: parse=1 empty=0 reward=0.765 lat~2s
fidw=0.0: parse=1 empty=0 reward=0.765 lat~4.7s  (ok, slightly slower)
fidw=1.0: parse=0 empty=3  << rejected
fidw=1.5: parse=0 empty=3  << rejected
Decision: keep default fidelity_loss_weight=0.5; increasing it hurts.

Captured: 2026-07-27T15:42:49.113431+00:00
