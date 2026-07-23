# E939-E998: role-safe decoding, aligned training, and bounded nesting

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
| E964 | E963 clean scratch / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 1.0000 | 0.4656 | 0.7500 | 0.9730 | 0 / 0 | 0/2 campaign |
| E964 | E963 clean scratch / held_out | 5 | 0.8000 | 0.4000 | 0.4000 | 0.4733 | 0.1404 | 0.4952 | 0.6540 | 1 / 5 | 0/2 campaign |
| E965 | E951 global binder-symbol reservation / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E965 | E951 global binder-symbol reservation / held_out | 5 | 0.8000 | 0.6000 | 0.6000 | 0.6500 | 0.3977 | 0.6000 | 0.7010 | 1 / 2 | 0/2 campaign |
| E968 | E951 fail-closed training boundary v256 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E968 | E951 fail-closed training boundary v256 / held_out | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/2 campaign |
| E973 | E972 weighted-mixture scratch / smoke | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5978 | 0.8333 | 0.9730 | 0 / 0 | 0/2 campaign |
| E973 | E972 weighted-mixture scratch / held_out | 5 | 1.0000 | 0.8000 | 0.4000 | 0.8333 | 0.3961 | 0.8286 | 0.9152 | 0 / 2 | 0/2 campaign |
| E974 | E951 visible-reference penalty 4 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E974 | E951 visible-reference penalty 4 / held_out | 5 | 1.0000 | 0.6000 | 0.6000 | 0.6833 | 0.4207 | 0.6286 | 0.8324 | 0 / 4 | 0/2 campaign |
| E975 | E951 visible-reference penalty 1 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E975 | E951 visible-reference penalty 1 / held_out | 5 | 1.0000 | 0.6000 | 0.6000 | 0.6833 | 0.4207 | 0.6286 | 0.8324 | 0 / 4 | 0/2 campaign |
| E976 | E951 withdrawn penalty v259 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E976 | E951 withdrawn penalty v259 / held_out | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/2 campaign |
| E977 | E951 typed-slot reservation v260 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E977 | E951 typed-slot reservation v260 / held_out | 5 | 1.0000 | 0.6000 | 0.6000 | 0.7333 | 0.4354 | 0.6286 | 0.8534 | 0 / 3 | 0/2 campaign |
| E978 | E951 symbol-slot reservation v261 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E978 | E951 symbol-slot reservation v261 / held_out | 5 | 1.0000 | 0.6000 | 0.6000 | 0.6833 | 0.4207 | 0.6286 | 0.8324 | 0 / 4 | 0/2 campaign |
| E979 | E951 withdrawn reservation v262 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E979 | E951 withdrawn reservation v262 / held_out | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/2 campaign |
| E981 | E980 binder-arity weight 1 / smoke | 3 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.3833 | 0.5000 | 0.6407 | 1 / 0 | invalid partial campaign |
| E982 | E980 binder-arity weight 1 / held_out | 5 | 1.0000 | 0.8000 | 0.4000 | 0.8833 | 0.5038 | 0.7190 | 0.9284 | 0 / 0 | 0/1 |
| E983 | E980 binder-arity weight 0 / smoke | 3 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.3478 | 0.5000 | 0.6487 | 1 / 0 | 0/1 |
| E984 | E980 binder-arity weight 0 / held_out | 5 | 0.8000 | 0.6000 | 0.2000 | 0.6333 | 0.2755 | 0.5286 | 0.7224 | 1 / 2 | 0/1 |
| E989 | E988 binder-arity weight 1 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.7500 | 0.5492 | 0.6667 | 0.8820 | 0 / 0 | 0/1 |
| E990 | E988 binder-arity weight 1 / held_out | 5 | 1.0000 | 0.6000 | 0.2000 | 0.6367 | 0.5030 | 0.5524 | 0.8364 | 0 / 2 | 0/1 |
| E991 | E980 arity 1 + schema types / smoke | 3 | 1.0000 | 0.6667 | 0.6667 | 0.7500 | 0.4347 | 0.5833 | 0.8680 | 0 / 2 | 0/2 campaign |
| E991 | E980 arity 1 + schema types / held_out | 5 | 0.8000 | 0.8000 | 0.6000 | 0.8000 | 0.4748 | 0.8000 | 0.7736 | 1 / 0 | 0/2 campaign |
| E992 | E951 unique binders v263 / smoke | 3 | 1.0000 | 1.0000 | 0.3333 | 0.7222 | 0.6573 | 0.5000 | 0.8537 | 0 / 1 | 0/2 campaign |
| E992 | E951 unique binders v263 / held_out | 5 | 1.0000 | 0.8000 | 0.4000 | 0.6667 | 0.5554 | 0.6952 | 0.8310 | 0 / 4 | 0/2 campaign |
| E993 | E951 withdrawal v264 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E993 | E951 withdrawal v264 / held_out | 5 | 0.8000 | 0.6000 | 0.6000 | 0.6333 | 0.3914 | 0.5619 | 0.6960 | 1 / 2 | 0/2 campaign |
| E994 | E951 withdrawal v264 / held_out retry | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/1 |
| E995 | E951 scoped binder parents v265 / smoke | 3 | 1.0000 | 1.0000 | 0.6667 | 0.8333 | 0.6518 | 0.6667 | 0.8910 | 0 / 0 | 0/2 campaign |
| E995 | E951 scoped binder parents v265 / held_out | 5 | 0.8000 | 0.6000 | 0.6000 | 0.6333 | 0.3914 | 0.5619 | 0.6960 | 1 / 2 | 0/2 campaign |
| E996 | E951 scoped-filter withdrawal v266 / held_out | 5 | 1.0000 | 0.8000 | 0.8000 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 | 0/1 |
| E997 | E980 arity + schema + lattice 2 / smoke | 3 | 0.6667 | 0.6667 | 0.6667 | 0.6667 | 0.3833 | 0.5000 | 0.6407 | 1 / 0 | 0/2 campaign |
| E997 | E980 arity + schema + lattice 2 / held_out | 5 | 0.8000 | 0.8000 | 0.6000 | 0.8000 | 0.4748 | 0.8000 | 0.7736 | 1 / 0 | 0/2 campaign |
| E998 | E980 Form-only 60s diagnostic | 1 | 1.0000 | 0.0000 | 0.0000 | 0.1667 | 0.1148 | 0.1429 | 0.6570 | 0 / 2 | 0/1 |

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

E963 completes 500 CPU scratch steps on E937 in 34.57 seconds with no inherited
weights or replay, final loss 3.1246, and local-only checkpoint SHA
`b795f362…e37df`. E964 improves smoke fidelity/reward to 1.0/0.973, but held
parse is 0.8, meaning/strict 0.4, fidelity 0.4733, structure 0.1404, recall
0.4952, reward 0.654, with one timeout and five fallback events. Reject E963;
never sync, promote, serve, resume, or use it as a parent.

E965 reserves remaining content symbols for every unresolved schema-typed
forward binder. It removes one of E957's two compiler dead ends, but the
coarse reservation changes the Tabs trajectory into a timeout and repeats
E962's held regression: strict 0.6, fidelity 0.65, structure 0.3977, and reward
0.701. Optional and nested component content has variable symbol cost, so a
global one-symbol-per-binder budget is not authoritative. Revert to v255.

E966 tests single-parent binder ownership after auditing 2,505 JSONL artifact
targets under the E937/E938 directories, including governance and rejection
artifacts, and finding zero multi-parent binders. The constraint also removes every
nontrivial supervised row from the optional binder-topology head on canonical
tree targets, so it fails focused tests before evaluation. Revert the decoder
change rather than silently disabling that training lever.

E967 closes the leak the audit exposed instead. Repository loaders were already
strict, but direct TwoTower and grammar-diffusion construction/training checked
only the older symbol-only rule. Both model boundaries now also reject named
markers, semantic-role prompt labels, and role-unsafe strings. The OpenUI pack
generator canonicalizes its emitted markers before returning records, and all
training fixtures use opaque binders and `:slot_<ordinal>` targets. A durable
test loads all 524 active E937 train records and all 50 E938 eval records; the
earlier primary-plus-alternate audit remains 632/632. The literal `"email"`
appears only as the official `Input.type` enum in the accepted test; the
role-unsafe `Slider.name` negative fixture now uses schema atom `"row"` and is
required to fail before model access.

E968 re-evaluates E951 under the retained v256 training boundary. Every smoke
and held quality aggregate is identical to E957, including held parse 1.0,
strict 0.8, fidelity 0.8333, structure 0.4434, recall 0.6952, reward 0.8834,
zero timeouts, and three certified fallbacks. The boundary is decode-neutral.
AgentV remains 0/2; E951 remains diagnostic-only because its weights descend
from pre-role-safe E891.

E969 fails before evaluation because visible-reference completeness remains a
choice-codec-only capability and cannot observe lexer `bN` binders. E970 is
interrupted at 559/1,200 before checkpoint finalization. E971 reaches 500/500
weighted-mixture steps but is interrupted during finalization and also has no
checkpoint. Neither interrupted run is evidence or resumable lineage.

E972 completes 450 clean scratch steps with the committed E937 mixture in 33.07
seconds, final loss 4.3559 (last-20 example mean 3.4385), and local-only SHA
`a905a6af…ccd742`. E973 improves substantially over uniform E963: held parse
1.0, fidelity 0.8333, recall 0.8286, reward 0.9152, zero timeouts, and two
fallbacks. However, strict meaning is only 0.4 versus E968's 0.8, and the Tabs
output still shares binders across parents. Reject E972; never sync, promote,
serve, resume, or use it as a parent.

E974 extends the existing visible-reference lever to lexer compiler paths as a
soft penalty on already-referenced binders; it leaves fresh references and
container closure neutral, so legal DAG supervision remains available. At
weight 4 it activates 20 times and changes three held-out choices, but
overcorrects: held strict falls from E968's 0.8 to 0.6, fidelity to 0.6833,
recall to 0.6286, reward to 0.8324, and fallbacks rise from three to four. The
Tabs and Form cases collapse to a one-slot TextContent fallback. Reject weight
4; do not weaken strict meaning or promote the treatment.

E975 lowers the same penalty to 1. It activates 13 times and changes only one
held-out choice, but every aggregate and fallback count is identical to E974.
The first choice flip is already harmful, so this is not a useful monotonic
weight sweep. Reject the generic reuse penalty rather than tune it further.

E976 evaluates the withdrawn treatment under v259. Every smoke and held-out
aggregate is exactly equal to E968, including held strict 0.8, fidelity 0.8333,
structure 0.4434, recall 0.6952, reward 0.8834, zero timeouts, and three
fallbacks. The rollback is decode-neutral while the opaque training-fixture
cleanup remains retained.

E977 reserves required content capacity for undeclared typed binders before
admitting another component with a required direct symbol. It preserves the
smoke baseline and three held fallbacks, but optional CardHeader properties can
still consume the reserved symbols. Form still reaches a compiler dead end,
Tabs loses two slots, and held strict regresses from 0.8 to 0.6. Reject v260 as
incomplete; reservation must apply where symbol tokens are consumed.

E978 applies the reservation at actual symbol-token consumption. Form and Tabs
both collapse to the one-slot certified fallback; held strict remains 0.6,
fidelity falls to 0.6833, recall to 0.6286, reward to 0.8324, and fallbacks rise
to four. Reject and withdraw the complete reservation treatment; constraining
downstream capacity changes the earlier compiler trajectory without resolving
the underlying binder overcommit.

E979 evaluates the complete withdrawal under v262. Every aggregate exactly
matches E976 and E968, proving the rollback restored the retained decoder. The
next treatment must address the upstream binder-arity choice rather than
reserve downstream symbols.

E980 tests that upstream hypothesis directly with 450 clean weighted E937
scratch steps, binder-arity loss weight 1, and no parent checkpoint. It
finishes in 36.93 seconds at loss 4.6926 and writes local-only SHA
`76a2b78d...14bb0`. E981 is an invalid interrupted campaign: only smoke
completed, one decode timed out, and no scoreboard or AgentV campaign bundle
was finalized. The bounded E982 held-only rerun with arity decode weight 1
reaches parse 1.0, fidelity 0.8833, structure 0.5038, and reward 0.9284, but
strict-v2 remains 0.4 and component recall falls to 0.7190; Tabs still reuses
`b2` across parents and Form covers only four of six slots. E983-E984 disable
the decode weight to isolate the auxiliary training effect: smoke still times
out, and held parse/strict/reward fall to 0.8/0.2/0.7224. The head supplies a
real held-out ranking signal, but the checkpoint is globally worse than E979.
E981-E990 omitted the schema-component-type switch used by E973/E979, so those
runs remain valid diagnostics but are not matched comparisons. E991 corrects that: held
strict rises to 0.6 and recall to 0.8, but parse falls to 0.8 with one timeout,
reward falls to 0.7736, and smoke meaning/fidelity regress. Reject E980; never
sync, promote, serve, resume, or use it as a parent.

E985-E987 are invalid interrupted attempts at binder-arity loss weight 0.25:
they stop at 397/450, 184/350, and 27/150 respectively without a checkpoint or
summary. The varying stop points came from orphaned command sessions, not the
harness wall cap; none is evidence or resumable. E988 keeps one persistent
terminal session and completes 150 clean steps in 70.54 seconds at loss 6.3801,
writing local-only SHA `36f57b3b...c92bf2`. E989-E990 evaluate arity decode
weight 1. Smoke recovers parse 1.0 but reaches only strict 0.6667 and fidelity
0.75. Held parse is 1.0, but strict falls to 0.2, fidelity to 0.6367, recall to
0.5524, and two fallbacks remain; Form and Tabs still share binders across
parents. Reject E988 as undertrained and dominated by E980/E979; never sync,
promote, serve, resume, or use it as a parent.

E992 tests decoder-only global binder-reference uniqueness without touching
gold-decision extraction or auxiliary supervision. It removes cross-parent
reuse but collapses Form to a one-slot fallback and Tabs to an empty array:
held strict falls 0.8 to 0.4, fidelity to 0.6667, reward to 0.8310, and
fallbacks rise from three to four; smoke strict falls to 0.3333. Reject and
withdraw v263. E993's smoke is exact E979 parity, while one transient held
timeout prevents campaign parity. The isolated E994 held retry exactly matches
all E979 aggregates and three fallbacks, establishing v264 rollback parity.

E995 narrows global uniqueness to lexical component-parent stacks: same-parent
references remain legal, while `Stack` to nested `Tabs` reuse is filtered. It
preserves smoke exactly but makes Tabs time out; held parse falls to 0.8,
strict to 0.6, fidelity to 0.6333, recall to 0.5619, and reward to 0.696.
Lexical parent identity still lacks enough semantic information at the choice
point. Reject and withdraw v265. E996 exactly matches E994/E979 held metrics
and three fallbacks, establishing v266 rollback parity.

E997 applies width-2 bottom-triggered lattice search to the schema-matched E980
arity arm. Held-out is exactly equal to E991, including its timeout, while
smoke adds a timeout and falls to parse 0.6667 and reward 0.6407. Search does
not recover the compiler dead end and worsens runtime stability; reject it.

E998 gives E991's timed-out Form row a 60-second diagnostic budget. It
finishes in 11.91 seconds only by emitting `root = TextContent(":slot_0")`:
one of six slots, one of seven component types, strict meaning 0, and two
fallbacks. The short result is not semantically equivalent minification; its
low fidelity and recall are correct. Raising the timeout does not repair E980.
