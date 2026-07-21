# E645 — `smoke_hero_01`'s regression: a mechanism match and a net-positive fix (variadic-frame gate)

Date: 2026-07-21
Status: mechanism-matched trace + scoped fix; net positive on one fresh scratch
checkpoint; not a confirmed identity with the original report; not a ship claim

## Which open item this picks

E630/E643/E644 have repeatedly flagged the same deferral: `smoke_hero_01`'s
regression (`True -> False`, `empty_root_stack`, first surfaced in E630's
widened-suite re-verification of E643's own schema-position-gate fix) "remains
untraced." This session investigates it directly.

## Environment note (read first)

The exact checkpoint E630/E643/E644 traced
(`outputs/runs/e626-required-slot-margin-scratch800-20260720/checkpoints/last.pt`,
sha256 `c5b7c807…dd561221`) does not exist in this session's filesystem —
`outputs/` is untracked and ephemeral across sessions, and prior iterations'
"reused verbatim" claims only held within a single continuous session. A fresh
scratch checkpoint was trained using E626/E620's exact documented recipe and
seed (E530-r2 corpus, 244 records, `context_backend=scratch`,
`output_tokenizer=choice`, seed 0, batch size 1, 800 steps,
`--no-sync-checkpoints`). Its `last_loss` (4.062225) is close to but **not**
identical to E626/E639's (4.068013) — consistent with environment/torch-version
nondeterminism across sessions (this session installed a fresh `.venv`,
`torch==2.5.1`), not a faithful bit-identical replay. Checkpoint:
`outputs/runs/e645-smoke-hero-trace-scratch800-20260721/checkpoints/last.pt`,
sha256 `a4c24987d1cf3dca97f84fd19e64cfc94210de27b75dc27cc4f7eb5f2041b7e7`, local
only, not synced, not promoted.

## Part 1: the exact regression did not reproduce

Before any code change, a margin sweep `{0, 1, 2, 3, 4, 6}` was replayed on
this fresh checkpoint against the `smoke` suite (`n=3`) with E630's exact
matched recipe. `smoke_hero_01`'s `meaningful_program_v1` stayed **`True`** at
every margin tested — the `True -> False` (`empty_root_stack`) flip
E630/E643/E644 reported did not reproduce on this checkpoint. This is not
itself surprising: greedy decode is path-sensitive, and this lineage has
already established (E630, E643) that several of these 19 records are
dominated by which-bias-fires-first fragility on an undertrained 800-step
scratch checkpoint. A non-reproduction on a *different* checkpoint neither
confirms nor refutes the original report — it means this session could not
lean on "the record now fails" and instead had to read the decode mechanism
directly.

## Part 2: the trace finds the mechanism anyway

Reading `decode_stats.constrained_selection_traces`
(`phase == "required_slot_margin"`, the E627/E640 instrumentation, already
surfaced end to end through `scripts.evaluate_model`'s per-suite eval JSON) for
`smoke_hero_01` at margin=2: 7 fires total, 1 flagged
`hijacked_non_slot_candidate=true`, at **position 14, `frame_depth=1`,
`frame_path=[{kind: "variadic", expr_type: "array", arg_index: 0}]`**
(`aggregation_scope: "structural_root_list"` — literally `root`'s own
`Stack([...])` children list, matching this record's gold structure
`root = Stack([title, rule, hero], "column")`). The pre-bias argmax was
`+TextContent` (`score_before=16.148`, a real component-opening candidate that
would otherwise have been chosen); the bias's floor
(`old_max + margin - target_score = 5.661`) pushed a slot-fill token (`@2`,
`score_after=18.148`) above it, replacing a real component with a bare slot
reference at an array-item position.

This is the *same class* of mechanism E628 already fixed for
`frame_depth == 0` and E630 already fixed for `component`/`object` argument
positions: the grammar legally allows a bare visible-slot token at the exact
same decode position as opening a real component, so
`_required_slot_margin_bias`'s floor (constructed to always exceed the current
max) wins there once schema-eligible, at any margin > 0. **E630's own gate
explicitly did not cover this position.** Its docstring stated `variadic`
frames are "left permissive… since this bias's only observed failure mode is
stuffing missing slots into optional enum/opaque *component*/*object*
properties, not array items" — an assumption, not itself traced. This
session's trace falsifies it directly.

This is a **mechanism match, not a proven identity** with the original
`smoke_hero_01` report — that checkpoint is unavailable to re-trace. It is the
most concrete, evidence-grounded explanation produced for this open item to
date, offered honestly as circumstantial, not certain.

## Part 3: fix

`TwoTowerModel._required_slot_margin_position_accepts_slot`
(`src/slm_training/models/twotower.py`) now gates `variadic` frames the same
way it already gated `object` frames: using the array's own item schema
(`frame.schemas[0]`, the tokenizer's own `ChoiceDecodeState._active_schema`
convention for variadic frames) through the existing
`_schema_can_reach_visible_slot` helper. An untyped/heterogeneous array
(`schemas` empty — meaning any expression, including a real component, is
grammar-legal there) or a typed item schema that cannot reach a visible slot
now gates the bias closed; a genuinely slot-reachable typed-array item schema
still fires unchanged. `component`/`object` gating (E630) and `frame_depth==0`
exclusion (E628) are untouched. Default (`0.0`) and the E617
`slot_contract_constrained_decode` contract-gated guard are unchanged.
`model.twotower` v80 → v81.

`tests/test_models/test_compiler_decode.py::test_required_slot_margin_bias_excludes_non_content_schema_positions`
(the pre-existing variadic case, which asserted the old permissive behavior
for an empty `schemas` tuple, now asserts the gated no-fire) plus a new
`test_required_slot_margin_bias_excludes_variadic_positions` (no item schema →
no fire; item schema that cannot reach a visible slot → no fire; item schema
that can reach a visible slot → fires unchanged). 147/147 passed across
`test_compiler_decode.py` + `test_choice_tokenizer.py` + `test_decode_stats.py`
(`PYTHONPATH=src NODE_OPTIONS="--max-old-space-size=8192" python -m pytest`).
`python -m scripts.verify_version_stamps --check`: ok after the `versions.json`
bump.

## Part 4: re-verification

Re-ran the fixed code against the same n=19 union (`held_out`=5, `rico_held`=3,
`adversarial`=4, `ood`=4, `smoke`=3, matching E642/E643/E644's own suite union,
still below `DEFAULT_MIN_SUITE_N=20` per suite) sweeping
`required_slot_margin_decode_weight` over `{0, 1, 2}`.

| Metric (pooled, n=19) | margin=0 | margin=1/2 |
| --- | ---: | ---: |
| meaningful_program_v1 rate | 0.7895 (15/19) | **0.8421 (16/19)** |
| reward_score (mean) | 0.7990 | 0.8107 |
| placeholder_fidelity (mean) | 0.6232 | 0.6645 |
| structural_similarity (mean) | 0.4521 | 0.5001 |
| component_type_recall (mean) | 0.6842 | 0.6579 |

`smoke_hero_01` stays `True` at every margin — its own `required_slot_margin`
trace at margin=2 confirms the traced position-14 `+TextContent` hijack is
gone (only slot-vs-slot swaps and one slot-vs-bind-reference hijack remain, see
below). **This is a net gain, not flat or negative** — every prior
re-verification in this lineage (E629, E630, E631) found a flat-or-negative
pooled effect after closing an over-stuffing gap; this is the first
net-positive result. Exactly one record differs from control at margin>=1:
`held_out_dual_card_01` (`False -> True`), unrelated to the traced
variadic-frame mechanism itself — margin=0's own prediction for that record is
a degenerate 44-item `Card` that already fails E644's `empty_children:Carousel`
check; margin>=1 produces a clean 5-item `Card` instead.

No regression on either previously-fixed record: `rico_eval_test_25` (E643)
stays `True` at every margin, and `ood_dashboard_01` (E643's revert of
E639/E641's gain) stays byte-identical to control across margins, exactly as
it already was pre-fix.

**A residual, narrower hijack class remains.** Post-fix, `hijacked_non_slot_
candidate=true` fires still occur at variadic-frame positions whose item
schema *is* slot-reachable (so the new gate correctly leaves them active), but
where the pre-bias argmax happened to be a bind-reference token (`kind=
"bind"`, e.g. `&2`) rather than a component-opening token. Counts across the
n=19 union at margin=2: `held_out` 1/14, `rico_held` 4/16, `smoke` 1/7,
`adversarial` 0/6, `ood` 0/12. This is a materially different (redirecting a
reference, not stealing a real component) and so far non-catastrophic failure
class on this checkpoint — not fixed or scoped here, flagged as the next open
item on this specific lever.

## Honest verdict

The specific `True -> False` (`empty_root_stack`) flip E630/E643/E644 reported
for `smoke_hero_01` did **not** reproduce on this session's freshly-trained
(same-recipe, non-identical) checkpoint, so this is not a confirmed
root-cause-and-fix of that exact prior observation — the checkpoint that
produced it is gone. What this session does establish with direct trace
evidence: E630's own schema-position gate left an identical-in-kind
vulnerability open at `variadic` array-item positions (an untested assumption
in its own docstring), it fires and hijacks a real component candidate on
`smoke_hero_01` itself on this checkpoint, and closing it — mirroring E630's
own fix pattern, extended via the array's item schema — is a net positive on
the widened n=19 suite, the first such result in this lineage's repeated
re-verification pattern. Single scratch checkpoint (800 steps, seed 0), n=19
below `DEFAULT_MIN_SUITE_N=20` per suite; not a ship claim; no checkpoint
trained, promoted, or synced; default `required_slot_margin_decode_weight`
(`0.0`) unchanged.

## Next step (deferred)

1. A residual, narrower hijack class (variadic-frame slot-vs-bind-reference,
   not slot-vs-component) still fires post-fix and was not root-caused here —
   worth its own trace if this lever is pursued further.
2. E620's original coverage-aware component/property closure recommendation
   remains the next lever this lineage has not directly tried.
3. A genuinely powered multi-seed/retrained comparison of
   `required_slot_margin_decode_weight` remains open (E626's own deferral,
   still not picked up).
4. If checkpoints persist across sessions in the future, a direct re-trace of
   E630/E643/E644's exact original checkpoint would let this finding be
   *confirmed* (not just mechanism-matched) against the original report.

Raw evidence:
[JSON](iter-e645-required-slot-margin-variadic-gate-20260721.json).
