# E627 — root-causing why `required_slot_margin_decode_weight=6` hijacks Dashboard's root

Date: 2026-07-20
Status: instrumentation + real single-record causal trace on E626's own scratch
checkpoint; mechanism identified with concrete decode-level evidence; not a
quality/ship claim (see "What this is not" below)

## Which E626 next step this picks

E626 deferred two follow-ups: (a) a powered multi-seed margin sweep, and (b) a
root-cause trace of *why* `required_slot_margin_decode_weight=6` regresses
structure (Dashboard collapses to a bare `Button`, Gallery's typed array
re-closes empty) while `=2` helps. This iteration picks **(b)**: it is
tractable within the run cap using E626's own already-trained scratch
checkpoint (no retrain needed) and answers E626's own framing of the question
directly: *"Is the margin large enough to override the initial
root-component-choice logit entirely ... or is it correctly scoped but the
model's own base logits for those early decode positions are so weak that any
large floor value dominates regardless of scope?"*

## Instrumentation added

`TwoTowerModel._required_slot_margin_bias` (E626) was not changed. Two
observability-only additions let the existing mechanism be inspected instead
of guessed at:

- `DecodeStats.required_slot_margin_applications` /
  `required_slot_margin_choice_changes` (`src/slm_training/models/decode_stats.py`),
  following the exact `slot_component_applications` /
  `slot_component_choice_changes` counter convention already used for
  `_schema_role_slot_bias`.
- `TwoTowerModel._record_required_slot_margin_trace`
  (`src/slm_training/models/twotower.py`), following the existing
  `_record_semantic_plan_root_trace` / `_record_semantic_plan_seed_trace`
  pattern: a bounded (shared 64-entry cap) `constrained_selection_traces`
  entry per fire, recording `frame_depth` (via the existing
  `_choice_phase_evidence` helper — `0` means no component frame is open yet,
  i.e. the position is a fresh top-level statement's value, not an argument
  inside an already-opened component), the pre-/post-bias argmax token *and
  its grammar `kind`* (`component_root_or_bound`, `component_bound`, `sym`
  for a visible-slot token, etc.), and a `hijacked_non_slot_candidate` flag
  (true iff the bias flips the argmax away from a non-slot candidate, i.e. a
  real component/structural token was outscored by a slot-fill token at that
  exact position).
- No bias formula, weight, or default changed — `model.twotower` stays v60
  with a `no-bump:` history entry (behavior-neutral instrumentation only);
  `python -m scripts.verify_version_stamps --check`: ok.
- Unit tests: `tests/test_models/test_compiler_decode.py::test_required_slot_margin_trace_flags_a_root_level_component_hijack`
  (asserts the trace correctly flags a `frame_depth=0` component-vs-slot
  hijack and correctly does *not* flag a `frame_depth=1` slot-vs-slot swap
  inside an already-open component) and
  `tests/test_models/test_decode_stats.py::test_decode_stats_aggregates_required_slot_margin_counts`.

## Experiment: single-record causal trace on E626's real scratch checkpoint

Reused E626's already-trained checkpoint verbatim (no retrain):
`outputs/runs/e626-required-slot-margin-scratch800-20260720/checkpoints/last.pt`,
sha256 `c5b7c807…dd561221` (local-only, matches E626's JSON exactly). Ran
`scripts.evaluate_model` on the `ood` suite with `--eval-limit 1` (record 0 =
`ood_dashboard_01`, the exact record E626 reported collapsing) and E626's full
matched-recipe flags, varying only `--required-slot-margin-decode-weight` over
`{0, 2, 6}`. All three runs reproduce E626's own per-record predictions
exactly for this record: margin 0 -> `Card([], ":ood.dash.status.body")`,
margin 2 -> the full 4-component `Stack`, margin 6 -> `Button(":ood.dash.m2.value")`
— confirming a faithful replay before reading the new traces.

### The mechanism, read directly off the traces

At margin=6, `required_slot_margin_applications_sum=5` (not the ~9 one would
expect if it fired throughout decode) — it fires exactly 5 times, once per
required slot, **all five at `frame_depth=0`, in the first 5 decode
positions**, before any component frame is ever opened:

| pos | frame_depth | pre-bias argmax (kind) | post-bias argmax | hijacked? |
| --- | --- | --- | --- | --- |
| 1 | 0 | `+Image` (component_root_or_bound) | `@3` (sym) | yes |
| 2 | 0 | `+Callout` (component_bound) | `@0` (sym) | yes |
| 3 | 0 | `@1` (sym) | `@1` (sym, unchanged) | no |
| 4 | 0 | `LIT_STR` (lit) | `@2` (sym) | yes |
| 5 | 0 | `@1` (sym) | `@4` (sym) | no (slot-vs-slot) |

The grammar legally allows a bare visible-slot token (`@N`) as a complete
top-level statement value — an alternative production to opening a real
component. `_required_slot_margin_bias`'s "still missing anywhere in the
prefix" criterion is true for *every* required slot simultaneously at decode
start, so it fires with maximum force at the very first few statement
positions, and a margin of 6 is large enough to floor the slot candidate
**above whatever real component the model would otherwise have opened there**
(`+Image` at 11.83 vs the floored `@3` at 17.83; `+Callout` at 16.26 vs floored
`@0` at 22.26). By position 6 every visible slot has appeared in the prefix,
so the lever goes permanently silent for the rest of the ~150-token decode —
but the damage is structural and irreversible: whichever statement is later
serialized as the program's `root` was decided in that already-corrupted
first stretch (or, for Dashboard, immediately follows it at position 6 with
`+Button`, itself only large enough to grab one leftover slot before the
program's other, later-built, real components — `+Callout`, two `+Card`s, a
`+Stack` — end up as unreferenced dead statements the compiler drops).

At margin=2 the *same* competing production is legal at the *same* position 1
(`+Image` at pre-bias argmax vs a floored `@3` candidate) — but the floor
(`old_max + 2`) is not large enough to survive the correcting biases that run
later in the same per-position stack (`semantic_plan_bias`, including
`semantic_plan_root_margin_decode_weight=2` / `semantic_plan_root_decode_weight=8`,
both fixed and active in every arm): those additive biases, applied *after*
`_required_slot_margin_bias` in the decode loop's bias-stacking order
(`schema_*` -> `slot_coverage_close` -> **`required_slot_margin`** ->
`repeated_array_close` -> `typed_array_nonempty` -> `semantic_plan_*`), push
the score of the correct component candidate back above the floored slot
token before the position's final `argmax()` is taken. The margin=2 trace
shows exactly this: `required_slot_margin_applications_sum=7`,
`choice_changed` true at several individual positions (the bias *does* win
its own local comparison), yet the actually-*emitted* token at position 1 is
still `+Callout` (a real component) — the downstream correcting bias reversed
it before the position committed.

### Answering E626's question directly

Not "leaking into component selection" in the sense of the bias ever adding
score to a component token — it never does; `_required_slot_margin_bias` is
exactly as narrowly scoped as designed. The leak is structural: the grammar
offers "assign this statement a bare missing placeholder directly" as a
**legal alternative to opening a real component at the same decode position**,
and whether the *intended* narrow slot-filling behavior or this degenerate
pass-through wins is decided purely by whether `required_slot_margin_decode_weight`
exceeds the combined magnitude of the downstream corrective biases
(`semantic_plan_root_margin_decode_weight`, `semantic_plan_root_decode_weight`,
etc.) that run later in the same per-position stack and are supposed to keep
structure on track. At margin=2 (below that threshold here) the corrective
biases win the race; at margin=6 (above it) they lose, and losing at the
*first* few decode positions — before any real structure exists — is
catastrophic in a way losing later would not be, because top-level statement
bindings are one-shot in this grammar.

## What this is not

A single record, one scratch checkpoint, `n=1` per arm — a causal trace, not
a quality confirmation (E626's own `n=4` numbers remain the quality evidence
of record). No checkpoint trained, promoted, or synced. No gate touched, no
default changed. This does not by itself validate E626's next-step (1)
(powered multi-seed margin sweep) — that remains open.

## Decision

`required_slot_margin_decode_weight` stays default-off and margin=6 stays
rejected, unchanged from E626. The new, concrete finding: the risk is not
"too strong a bonus somewhere in the tree" in the abstract, it is specifically
**a race against the fixed downstream `semantic_plan_root*` biases at the
very first few decode positions**, so any future tuning of this margin should
be evaluated relative to those biases' magnitude (here 2 and 8), not in
isolation — and a cheap structural mitigation worth trying next is moving
`_required_slot_margin_bias` to run *after* `_semantic_plan_bias` in the
per-position stack (or excluding `frame_depth == 0` candidates from its
target set), which the trace evidence suggests should let it keep flooring
slots-as-arguments while no longer competing for the root/top-level statement
choice at all.

## Next step (deferred)

1. E626's original next-step (1): a powered multi-seed replay sweeping the
   margin against a full held-out/`rico_held` suite remains open and
   untouched by this iteration.
2. Try the two structural mitigations named above (reorder the bias stack, or
   exclude `frame_depth == 0` from `_required_slot_margin_bias`'s target set)
   and re-run this same single-record trace to confirm the hijack no longer
   fires at `frame_depth == 0` before considering either a real code change.
3. This trace only exercised Dashboard; Gallery's "typed array re-closes
   empty" failure mode (E626's other headline margin=6 regression) was not
   independently traced here and may or may not share this exact mechanism —
   worth a matching one-record trace before generalizing.

Raw evidence:
[JSON](iter-e627-required-slot-margin-root-cause-trace-20260720.json).
