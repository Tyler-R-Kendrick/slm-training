# E939-E952: role-safe string decoding and aligned training

E939 established that the E891 checkpoint still produced grammar-valid layouts
with weak topology on the role-audited E938 suites. E940's strict compiler-tree
control improved smoke strict-v2 to 1.0 and held-out strict-v2 to 0.2, but direct
inspection found request content slots in structural `name` properties. E941's
broader component-type constraint raised held-out strict-v2 to 0.4 while
regressing meaning, fidelity, recall, and fallback count, so it remains rejected.

The decoder defect was an unconditional union: content-slot token IDs were added
to every schema string property. v250-v252 make the property role authoritative.
Content properties may use only request-local slots; structural identifier
properties may use only opaque `"$N"` atoms; unclassified open strings fail
closed when a slot contract is active. The same change adds identity-based
output-vocabulary remapping and shared-prefix context-position resizing so a
pre-change checkpoint can warm-start onto the expanded role-safe vocabulary.

| Run | Checkpoint / suite | n | parse | meaning-v1 | strict-v2 | fidelity | structure | recall | reward | timeout / fallback | AgentV |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E939 | E891 / smoke | 3 | 1.0000 | 0.6667 | 0.0000 | 1.0000 | 0.1785 | 0.4167 | 0.9900 | 0 / 0 | 0/2 campaign |
| E939 | E891 / held_out | 5 | 1.0000 | 0.2000 | 0.0000 | 1.0000 | 0.1310 | 0.2952 | 0.9778 | 0 / 0 | 0/2 campaign |
| E940 | E891 strict tree / smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5783 | 0.6667 | 0.9450 | 0 / 0 | 0/2 campaign |
| E940 | E891 strict tree / held_out | 5 | 0.8000 | 0.6000 | 0.2000 | 0.6667 | 0.3527 | 0.5143 | 0.7234 | 1 / 2 | 0/2 campaign |
| E941 | E891 schema types / held_out | 5 | 0.8000 | 0.4000 | 0.4000 | 0.5000 | 0.3390 | 0.4286 | 0.6524 | 1 / 4 | 0/2 campaign |
| E947 | E944 / held_out, 4s diagnostic | 5 | 0.4000 | 0.4000 | 0.2000 | 0.4000 | 0.1240 | 0.3000 | 0.3892 | 3 / 0 | 0/1 |
| E948 | E891 role-safe strings / smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5783 | 0.6667 | 0.9450 | 0 / 0 | 0/2 campaign |
| E948 | E891 role-safe strings / held_out | 5 | 0.8000 | 0.0000 | 0.0000 | 0.2500 | 0.0793 | 0.0952 | 0.5606 | 1 / 8 | 0/2 campaign |
| E952 | E951 role-safe warm start / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.5045 | 0.6667 | 0.8990 | 0 / 0 | 0/2 campaign |
| E952 | E951 role-safe warm start / held_out | 5 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.1874 | 4 / 0 | 0/2 campaign |

E942 (549/600) and E943 (439/480) hit the cumulative wall cap before checkpoint
finalization and are invalid. E945 completed only smoke before campaign
interruption; E946 completed no suite. E949 failed before step 1 on output-vocab
and position-shape mismatches. E950 was a CLI parse failure and created no run.
None is evidence. E944 completed 350 scratch steps in 25.45 seconds, and E951
completed its 20-step E891 warm start in 2.91 seconds after the loader repair.

E948 and E952 generated zero request-slot-in-structural-property violations on
held-out rows; E952 had zero role-contract violations across all eight emitted
smoke/held predictions. That safety result does not offset quality failure.
E944 and E951 are local-only rejected checkpoints: do not sync, promote, serve,
resume, or use either as a parent. The role constraint is retained as a hard
correctness boundary; the next experiment must improve termination and topology
without relaxing it. No ship gate passed.
