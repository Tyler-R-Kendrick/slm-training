# E321 slot-owner balance audit — 2026-07-17

E321 audits only the committed E316 training corpus; no evaluation records or
IDs are used. Direct component owners are derived from parsed training ASTs.

The corpus contains 1,545 supervised slots across 22 component classes.
`TextContent` owns 799 (51.7%), followed by `Button` 121 (7.8%),
`ImageBlock` 72 (4.7%), `Callout` 62 (4.0%), and `Input` 36 (2.3%).
The E317/E318 final-20 slot accuracy of 70.1% is therefore only 18.4 points
above a majority-class predictor and can hide poor minority recall.

Training-only role counts show useful generalized signal already exists:
`email` maps to Input 22/28 times, `placeholder` to Input 28/28,
`subtitle` to CardHeader 11/11, `submit` to Button 19/24, and `continue` to
Button 8/8. The bottleneck is objective imbalance, not missing role vocabulary.

**Verdict:** do not add eval-shaped aliases. Retain E316 data and test a matched
focal slot-owner loss that downweights easy majority examples, while reporting
the majority baseline beside raw accuracy.
