# DSH3-26 operator termination

SLM-401 adds a default-off STOP supervision contract beside the sanitized
operator-policy objective. `OperatorTerminationTargetV1` keeps replay-only
remaining distance and proof IDs outside `model_input()`. STOP is a separate
control-plane class, never an `AstOperatorV1` or compiler action.

PARTIAL coverage may not supervise STOP: budget exhaustion and incomplete
enumeration route to fallback rather than terminality. COMPLETE singletons
retain compiler bypass authority; optional training masking changes only loss
weight, never inference behavior.

`operator_termination_losses` compares factored STOP/action and a joint
STOP-plus-known-action distribution. Selection uses Brier, ECE, edit-count TV,
premature/late stop, and reached-target rate—not training loss. This is a
unit-contract implementation; a bounded local corpus/matrix run remains
required before any quality claim.
