# E765-E780 — schema-closed constrained decoder

**Date:** 2026-07-22  
**Decision:** retain TwoTower v223 and eval harness v45; do not promote the checkpoint  
**Evidence:** [`iter-e765-e780-schema-closed-decoder-20260722.json`](iter-e765-e780-schema-closed-decoder-20260722.json)

E765 began from the E764 held-out failure: four of five records reached a
certified minimal fallback. The lattice probe exceeded its 12-second record
budget and is invalid evidence. E766's first symbol-exhaustion filter was a
no-effect regression. E767-E770 isolated two real compiler defects: postfix
operators remained admissible at component-reference array boundaries, and
empty node arrays could close. The retained schema-derived boundaries improved
the form row and made the input row fully request-bound without a fallback.

E771-E779 then repaired the settings failure at its source. Bounded compiler
telemetry showed `CardHeader(` was admitted after all declared markers were
consumed. The completion forest now derives from the official component
schema whether a candidate opens text or child-node content, requires a
nonempty typed array, admits a typed component/reference, or closes a binder
cycle. After marker exhaustion it admits only closed leaf nodes and certified
acyclic references. It therefore cannot select a component whose required
content is unavailable, recurse through containers until the canvas ends, or
defer an invalid typed/cyclic tree to finalization. Certified finalization
fallbacks are counted in canonical eval telemetry instead of being invisible.

The retained E780 local CPU replay (`held_out`, n=5) has parse, contract
precision/recall, marker fidelity, and marker validity all at 1.0, with zero
fallbacks and zero timeouts. Compared with E764, fidelity rises from 0.4019 to
1.0, structure from 0.2155 to 0.3921, and reward from 0.7336 to 0.9466 while
fallbacks fall from four to zero. This is not a ship result: meaningful-v1 is
0.2, strict meaningful-v2 is 0, component recall is 0.2571, and AgentV is 0/1.
The checkpoint still assigns several markers to the wrong component roles.

Every valid arm ran locally under the 110-second command guard and a 12-second
per-record decode guard. No remote workflow ran. No checkpoint was created,
synced, or promoted, so the model card is unchanged.

Changed-surface verification also found two stale free-form producer paths.
The OpenUI DSL pack now templatizes generated ProgramSpecs before emitting
records and asserts the symbol-only contract at that boundary. The canonical
settings train seed replaces its `notify` and `volume` literals with declared
markers. Harness train-data v11 records both producer repairs; no acceptance
gate or threshold changed.

The first PR Python job was cancelled by the unchanged two-minute job cap: a
cold dependency setup plus 320 serial changed tests exceeded the budget. The
same selected tests pass locally with four workers in 18.23 seconds in the
canonical `check_changed` harness, whose worker count is one named constant
(versus 33.34 seconds serial). Test selection and assertions are unchanged,
and the workflow requires no special configuration or dependency.
