# E832-E834: longer v4 baseline

## Outcome

E832 trained the target-inventory-correct v4 corpus for 600 local CPU steps in
63.30 seconds. Loss ended at 4.8127 after 616,443 prompt tokens and 132,926
target tokens. The scratch checkpoint SHA-256 is
`01561831ea4638dec9f29d7cea085152f471940579751c92ba1931db9477992d`.
It was explicitly not synced.

E833 smoke n=3 reached parse 1.0, meaning-v1 0.6667, fidelity 1.0,
structure 0.2494, component recall 0.4167, reward 0.9820, p95 8.56 seconds,
and zero fallback/timeout. E834 held-out n=5 reached parse 1.0, meaning-v1
0.2, fidelity 1.0, structure 0.2995, component recall 0.2952, reward 0.9682,
p95 8.68 seconds, and zero fallback/timeout. Strict-v2 and AgentV are 0 on
both runs.

## Comparison and diagnosis

Against the 120-step E830 held-out replay, longer training improves fidelity
0.8857 to 1.0 and reward 0.9195 to 0.9682, but structure falls 0.3523 to
0.2995 and component recall is unchanged at 0.2952. All three smoke rows and
all five held-out rows fail strict-v2 with `duplicate_subtree_spam`; four of
five held-out rows also miss prompt-required components.

## Decision

Reject more undirected steps as the next lever. Retain E832 only as a compatible
local diagnostic checkpoint. Do not sync, promote, deploy, or make a ship
claim. The next experiment should enforce generalized AST/reference diversity
and prompt-component coverage without using semantic marker labels.

