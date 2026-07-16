# E50 symbol fidelity duration control — 2026-07-15

The E50 recipe (`fidelity_loss_weight=4.0`, lexer symbol tables) was extended
from 256 to 512 CPU steps. The longer run regressed:

| metric | result |
| --- | ---: |
| parse | 0/3 |
| structural similarity | 0.2007 |
| placeholder fidelity/reward | 0 |
| p50 latency | 10.1s |

The 256-step E50 checkpoint remains the better bounded result (strict
structural similarity 0.4244). Reject duration-only extension; the next
intervention must target symbol closure/sequence behavior directly.

This is scratch smoke evidence, not a ship claim.
