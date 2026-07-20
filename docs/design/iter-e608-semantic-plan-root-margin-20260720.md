# E608 — verifier-approved semantic-plan root margin

Date: 2026-07-20
Status: completed and rejected for promotion

E608 adds a default-off margin for verifier-approved semantic-plan root and
completion tokens. The verifier and exact completion validation are unchanged.
With margin 2, the approved token is floored two points above the best legal
score; the existing root weight remains a lower bound.

The capped, matched OOD `n=4` replay completes normally. It improves
meaningful-v1 from 0.50 to 0.75 and structural similarity from 0.5756 to
0.6698. Dashboard and gallery now reach verified Stack roots, and auth now
accepts the first verified EOS instead of constructing a duplicate Stack.

The intervention fails its preregistered quality constraint. Strict meaning-v2
remains 0, placeholder validity falls from 0.755 to 0.660, and reward falls
from 0.8145 to 0.6788. Gallery still emits an empty `ImageGallery` and scores
zero reward. Dashboard reaches only Button and Callout in its canonical root,
still misses the required Card family, and retains schema/role and placeholder
spam failures. Modal and auth retain schema-value role mismatches.

Reject root margin 2 as a promotable policy. The experiment establishes that
verified root reachability was a real bottleneck, but the next iteration must
repair visible schema-value role selection and root reference completeness
without forcing low-quality closures.

No checkpoint was created, promoted, or synced.

Evidence: [JSON](iter-e608-semantic-plan-root-margin-20260720.json).
