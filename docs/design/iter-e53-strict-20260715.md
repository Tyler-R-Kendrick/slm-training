# E53 strict boundary-weight evaluation — 2026-07-15

Strict evaluation confirms E53's raw syntax improvement but rejects its quality
tradeoff:

| metric | result |
| --- | ---: |
| raw syntax validity | 2/3 |
| parse | 0/3 |
| structural similarity | 0.2844 |
| placeholder fidelity/reward | 0 |
| p50 latency | 6.7s |

The stronger boundary loss makes outputs closer to syntactic form while losing
component structure. E53 is rejected for promotion; E52's lower boundary weight
remains the better structural setting.

This is strict scratch smoke evidence, not a ship claim.
