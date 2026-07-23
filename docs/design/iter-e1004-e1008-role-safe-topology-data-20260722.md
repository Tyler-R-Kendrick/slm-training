# E1004-E1008 — role-safe topology data supplement

Date: 2026-07-22. All builds, training, and evaluations ran on CPU under the
repository wall cap. The checkpoint was a fresh scratch train with explicit
`--no-sync-checkpoints`; it is not a ship candidate or reusable parent.

## Finding

The suspicious natural-language-looking fixtures were not model supervision.
The historical role-invalid Slider-name row was an intentionally invalid
loader test and now uses a closed DSL atom as its rejected value. The
historical positive completion-forest fixture was corrected to use `$N`
structural identifiers and `:slot_N` content slots. A direct audit found zero
role violations across all 582 E937 primary/alternate targets and all 50 E938
targets.

E1004-E1005 tested the narrower data hypothesis that Form and Tabs failures came
from insufficient repeated-topology coverage. Both proposed source targets were
opaque before sanitization. E1004 correctly quarantined the first three-tab
shape at G11 because `TabItem.content` did not admit the proposed Card/Button
values. No gate was weakened. E1005 used schema-valid text panels and admitted
four variants of each new Form and Tabs topology.

The strict E1005 build admitted 532 of 1,486 collected candidates. All 590
primary/alternate targets passed the role audit. Its required quality artifacts
reported 26 recommendations and 26 experiment candidates. The pre-existing
warnings remained explicit: 102 eval-overlap candidates and 20
placeholder-contract violations were rejected, not admitted.

## Results

| Run | Suite / stage | n | parse | strict-v2 | fidelity | structure | recall | reward | timeout / fallback |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E1005 | strict data build | 532 rows | — | — | — | — | — | — | 590 targets, 0 role violations |
| E1006 | fresh scratch train | 450 steps | — | — | — | — | — | loss 3.8939 | 56.62s |
| E1007 | smoke | 3 | 1.0 | 0.6667 | 0.8333 | 0.6981 | 0.5833 | 0.8870 | 0 / 0 |
| E1008 | held_out | 5 | 0.8 | 0.4 | 0.5700 | 0.3965 | 0.6000 | 0.6746 | 1 / 2 |
| E996 baseline | held_out | 5 | 1.0 | 0.8 | 0.8333 | 0.4434 | 0.6952 | 0.8834 | 0 / 3 |

Both complete evals emitted AgentEvals JSONL and pinned AgentV bundles; each
failed its single evaluated gate row (`0/1`). E1007 improved smoke structural
similarity over E979's 0.6518, but E1008 regressed every decisive held metric.
The held Form timed out to an empty prediction and Tabs collapsed to
`root = TextContent(":slot_0")`.

## Decision

Reject E1006 and never sync, promote, serve, resume, or use it as a parent.
Withdraw both producer fixtures: repeating the same target through four prompt
variants perturbed the global mixture but did not teach reliable Form/Tabs
construction. Retain train-data v26 with a no-bump history note. The next
data-level experiment must target topology exposure without multiplying
identical targets through paraphrases.
