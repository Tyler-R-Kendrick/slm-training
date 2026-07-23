# E939-E962: role-safe decoding, aligned training, and bounded nesting

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
| E953 | E951 resolved-document termination / held_out | 5 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.2000 | 0.1874 | 4 / 0 | 0/2 campaign |
| E954 | E951 Form-only, 60s diagnostic | 1 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0800 | 0.4286 | 0.9430 | 0 / 0 | 0/1 |
| E955 | E951 depth-3 bound / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.5045 | 0.6667 | 0.8990 | 0 / 0 | 0/2 campaign |
| E955 | E951 depth-3 bound / held_out | 5 | 1.0000 | 0.8000 | 0.2000 | 0.7833 | 0.3876 | 0.6524 | 0.8936 | 0 / 0 | 0/2 campaign |
| E956 | E951 depth-3 + schema types / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E956 | E951 depth-3 + schema types / held_out | 5 | 1.0000 | 0.8000 | 0.6000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/2 campaign |
| E957 | E951 role-unique + schema types / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E957 | E951 role-unique + schema types / held_out | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/2 campaign |
| E958 | E951 inline typed items / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E958 | E951 inline typed items / held_out | 5 | 1.0000 | 0.6000 | 0.6000 | 0.6833 | 0.4207 | 0.6286 | 0.8324 | 0 / 4 | 0/2 campaign |
| E959 | E951 lattice-bottom width 2 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E959 | E951 lattice-bottom width 2 / held_out | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/2 campaign |
| E961 | E951 plan-owned array close / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E961 | E951 plan-owned array close / held_out | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/2 campaign |
| E962 | E951 compiler-native plan close / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E962 | E951 compiler-native plan close / held_out | 5 | 0.8000 | 0.6000 | 0.6000 | 0.6500 | 0.3977 | 0.6000 | 0.7010 | 1 / 2 | 0/2 campaign |

E942 (549/600) and E943 (439/480) hit the cumulative wall cap before checkpoint
finalization and are invalid. E945 completed only smoke before campaign
interruption; E946 completed no suite. E949 failed before step 1 on output-vocab
and position-shape mismatches. E950 was a CLI parse failure and created no run.
None is evidence. E944 completed 350 scratch steps in 25.45 seconds, and E951
completed its 20-step E891 warm start in 2.91 seconds after the loader repair.

E953 makes complete, reference-resolved documents terminate immediately. Its
predictions are byte-identical to E952, but successful held latency improves.
E954 then shows the timeout is a 193-symbol alternating container tree, not an
infinite loop. Across 551 structured E937 primary/alternate targets, inline
element depth has median 1, p95 2, and maximum 3. E955 enforces that observed
maximum by allowing leaf components and references after two open recursive
containers while rejecting another recursive wrapper.

E955 preserves zero role violations across all eight outputs and recovers every
held row: parse 0.2→1.0, timeouts 4→0, meaning-v1 0.2→0.8, fidelity
0.2→0.7833, structure 0.2→0.3876, recall 0.2→0.6524, and reward
0.1874→0.8936. Strict-v2 remains 0.2 and AgentV remains 0/2. Retain v253-v254,
but do not promote or sync E951: its weights descend from pre-role-safe E891 even
though the current decoder makes outputs role-safe. E944 remains the clean
scratch lineage. No ship gate passed.

E956 adds authoritative schema component types to the retained depth bound. It
raises held strict-v2 0.2→0.6, fidelity 0.7833→0.8333, structure
0.3876→0.4434, and recall 0.6524→0.6952 with zero timeouts. Reward slips
0.8936→0.8834 and certified fallback rises 0→3, so this is retained as a
decoder capability rather than a ship claim. Raw-position role auditing finds
zero violations across all eight predictions; the earlier apparent
`Stack.gap="center"` violation was an audit bug caused by stripping the style
argument before mapping positional properties.

E957 observes that none of E937's 582 targets reuse one opaque structural ID in
the same component/property role. v255 therefore rejects same-role reuse while
preserving cross-role identity relationships. The matched run is otherwise
byte-identical to E956, but changes the repeated `ImageBlock("$44")` to fresh
role-local IDs and raises held strict-v2 0.6→0.8. All eight outputs remain
role-safe; AgentV remains 0/2 and three fallback events remain, so no ship claim.

E958 prohibited fresh forward references for slot-consuming typed-array items,
forcing those items inline so their content cost was locally visible. It did
not rescue Form and collapsed the previously strict Tabs row to the certified
minimal fallback. Held meaning/strict fell 0.8→0.6, fidelity 0.8333→0.6833,
structure 0.4434→0.4207, recall 0.6952→0.6286, reward 0.8834→0.8324, and
fallback rose 3→4. The v256 treatment was reverted; retain v255/E957.

E959 applies the existing width-2 lattice only when greedy decoding reaches a
bottom. Held decoding performs 16 rollbacks and exhausts the search budget
twice, but Form still takes the same certified fallback and every quality
aggregate is unchanged from E957. Held p95 latency worsens 5252.52→6888.31 ms.
Reject the search treatment; earlier feasibility accounting is required.

E960 failed before evaluation because the CLI-exposed semantic-plan array-close
lever was registered as choice-codec-only. E961 temporarily exposed it to the
lexer compiler and extended ownership to single authored families, but records
zero array-close applications and is quality-identical to E957. The runtime
method still consumes choice-codec frames, so the apparent lexer path was a
no-op. Revert model v256 and lever registry v34; retain v255/v33.

E962 adds a real compiler-native close bias for arrays owned by a prompt-plan
component or a single prompt-planned item type. It times out Form with an empty
prediction and collapses Tabs, reducing held parse 1.0→0.8, strict 0.8→0.6,
fidelity 0.8333→0.65, structure 0.4434→0.3977, recall 0.6952→0.6, and reward
0.8834→0.701. Revert the treatment and keep E957/v255.
