# Rescue bad seeds with more steps @ lr=1e-3

**Honesty:** fixture_or_scratch. **Not ship.**

# Rescue bad seeds with more steps at lr=1e-3 ASAP t30 (NOT SHIP)
seed42 s16: parse=0.33 empty=2 (from champ multiseed)
seed42 s30(~21 wall): parse=0.33 empty=2
seed42 s36(~27 wall): parse=1.0 meanful=0.33 empty=0 reward=0.765 lat=4191  << RESCUED
seed51 s16: parse=0 empty=3
seed51 s30(~25 wall): parse=0.67 empty=1
seed51 s36(~31 wall): parse=0 empty=3  << non-monotonic collapse

Decision: for seed42, more steps at lr=1e-3 rescues quality; seed51 remains brittle. Multi-seed selection still required.

Captured: 2026-07-27T15:29:56.170282+00:00
