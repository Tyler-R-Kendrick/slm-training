# E48 atomic dead-end candidate telemetry — 2026-07-15

Dead-end telemetry now records candidate counts at the constrained repair
failure event. The E48 checkpoint remains 0/3 parse with 12 dead ends. The
candidate-count sum was 32 (10.667 per generated example), so the grammar is
not simply empty throughout the failing path.

This shifts the diagnosis toward an invalid candidate sequence or grammar
state progression before the final dead end. The next harness change should
persist the prefix and chosen-token trace around the first dead end, then the
next model/grammar intervention can be selected from that trace.

This is scratch smoke evidence, not a ship claim.
