# E839-E840: all-path semantic-plan margin

## Outcome

E839 experimentally broadened the existing semantic-plan margin from the best
legal component to the best legal grammar path. On smoke (`n=3`) it preserved
strict-v2 3/3 and opaque-marker fidelity 1.0, removed every final binder, raised
structure from E837's 0.6033 to 0.6572, and reduced p50 from 3.07 to 1.74
seconds. There were no timeouts or unconstrained fallbacks.

E840 rejected the change on the frozen held-out slice (`n=5`). Strict-v2 stayed
0, two records timed out, parse was 0.6, fidelity 0.3333, structure 0.1263,
component recall 0.1810, and reward 0.4822. These are all worse than E838 except
parse and timeout count, which merely tie. Traces show typed repeated-component
collections can still accumulate binder references before the incomplete plan
reaches their required item family.

## Decision

Reject the all-path margin and restore the component-only comparison in v231.
The problem is structured plan/schema reachability for repeated and typed
collections, not template-marker naming or model-side marker conversion. No
checkpoint, sync, deployment, promotion, or ship claim was produced.
