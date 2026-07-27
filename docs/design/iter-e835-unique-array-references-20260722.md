# E835: unique structural-array references

## Outcome

E835 replayed the unchanged local E832 checkpoint on the frozen held-out
subset (`n=5`) after making repeated binder references within one structural
array compiler-illegal. The rule is derived only from AST binder identity; it
does not inspect template-marker names, semantic labels, or free text.

The local CPU run reached parse 1.0, meaning-v1 0.2, fidelity 1.0, validity
0.6, structure 0.2493, component recall 0.2952, reward 0.9874, p50 5.41
seconds, and p95 5.61 seconds with zero timeout or fallback. AgentV remained
0/1 and strict-v2 remained 0/5.

## Diagnosis

The constraint removed repeated direct references such as `b1, b2, b1, b1`
from generated child arrays. It did not remove `duplicate_subtree_spam`: the
decoder independently constructed three or more identical closed-schema
subtrees. All five rows also remained missing prompt-required grammar
components. Relative to E834, reward improved from 0.9682 to 0.9874, structure
fell from 0.2995 to 0.2493, and component recall was unchanged.

## Decision

Retain the grammar invariant because repeated ownership of one bound AST node
within a structural array is invalid tree construction and no model should be
asked to learn around it. Reject E835 as a quality improvement or ship signal.
The next harness/data intervention must expose required component coverage as
grammar/AST symbols; it must not infer component ownership from template names
or ask the model to convert marker text.
