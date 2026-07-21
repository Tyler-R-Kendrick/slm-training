# E630 — root-causing rico_eval_test_25's frame_depth>=1 over-stuffing (and an honest negative it uncovers)

Date: 2026-07-20
Status: root-cause trace + scoped fix, verified against the exact failing
record; wider re-verification finds the fix also reverts E626/E628's
previously-reported gain and surfaces a third regression -- not a
confirmatory result in either direction; not a ship claim

## Which deferral this picks

E629 found a new failure mode at `required_slot_margin_decode_weight>=1`
distinct from E627/E628's `frame_depth==0` root hijack: on
`rico_eval_test_25`, one `Button` absorbed 5 required slots across all 5 of
its positional arguments, flipping `meaningful_program_v1` `True -> False`
and exactly offsetting `ood_dashboard_01`'s real gain on the pooled binary
gate. E629 explicitly deferred root-causing this the way E627 root-caused
the `frame_depth==0` hijack. This iteration does that.

## Root cause, read directly off the traces

Reused E626's own scratch checkpoint verbatim (sha256
`c5b7c807…dd561221` verified). E627's `DecodeStats.constrained_selection_traces`
instrumentation already surfaces every `required_slot_margin` fire through
`scripts.evaluate_model`'s per-suite eval JSON (`decode_stats
.constrained_selection_traces`) — no new plumbing was needed, only pointing
it at `rico_held --eval-limit 1` instead of `ood`.

Before any fix, replaying E629's matched recipe on this one record:

| | margin=0 (control) | margin=1 (treatment) |
| --- | --- | --- |
| prediction | `root = Stack([v0,v1,v2,v3,v4],"column")`; `v0=Button(":sliding_tabs.label")`; ... | `v0 = Button(":sliding_tabs.label", ":cardview.title", ":toolbar.text", ":cardview.body", ":cardview_1.title")`; ...; `v10 = Card([])` |
| `meaningful_program_v1` | True | **False** |
| failure reason | — | `empty_card` (the last Card, `v10`, ends up with zero children and nothing else) |

`required_slot_margin_applications_sum=9` at margin=1 on this record; 6 of 9
fires are `hijacked_non_slot_candidate=true`, and every one of those 6 lands
on a **schema-constrained enum or opaque argument position of an
already-open component** (`frame_depth` in `{1,3}`, exactly where E628
already scoped this bias to fire) — never the root:

| position | frame_depth | component.property | before | chosen | hijacked? |
| --- | --- | --- | --- | --- | --- |
| 2 | 1 | `Button.label` (content) | `@1` | `@1` | no — intended fire |
| 3 | 1 | `Button.action` (opaque) | `-` (struct) | `@2` | **yes** |
| 4 | 1 | `Button.variant` (enum) | `LIT_STR` | `@0` | **yes** |
| 5 | 1 | `Button.type` (enum) | `-` (struct) | `@3` | **yes** |
| 6 | 1 | `Button.size` (enum) | `LIT_STR` | `@4` | **yes** |
| 11 | 3 | `TextContent.text` (content) | `@2` | `@7` | no — slot swap |
| 12 | 3 | `TextContent.size` (enum) | `LIT_STR` | `@6` | **yes** |
| 19 | 1 | `Card.variant` (enum) | `-` (struct) | `@5` | **yes** |

`Button` alone absorbs its content property (`label`, correctly) *and* all
four of its other properties (`action`/`variant`/`type`/`size` — none of
them content-bearing), exactly matching E629's own description. The same
pattern recurs on `TextContent.size` and `Card.variant`.

**The mechanism is structurally different from E627/E628's.**
`_required_slot_margin_bias`'s floor —
`bias[target] = max(0, scores.max() + margin - scores[target])` — is
computed relative to the *current, already-biased* max score, i.e. **after**
`_schema_value_bias` (enum discouragement), `_schema_opaque_bias` (opaque
discouragement), `_schema_enum_close_bias` / `_schema_opaque_close_bias`
(closure preference) have already run in the same per-position stack
(`schema_value -> schema_enum_close -> schema_opaque -> schema_opaque_close
-> schema_role_slot -> slot_coverage_close -> required_slot_margin -> ...`).
Because the floor is built to exceed *whatever the max currently is*, it
wins against all four of those upstream biases at **any margin > 0**, not
only a large one — E627/E628's finding was a genuine magnitude race against
a *later* corrective bias (`semantic_plan_root*`); this is a **guaranteed
win against earlier discouragement biases**, by construction, limited only
by whether a slot candidate is legal at the position at all. And the
grammar's own `ChoiceDecodeState._schema_accepts` (`choice_tokenizer.py`)
legally accepts a placeholder at *any* string-typed argument regardless of
enum/opaque-ness (`expected == "string" and expr_type == "placeholder":
return True`, no enum/opaque carve-out) — so nothing upstream can out-argue
it once it fires.

## The fix

`TwoTowerModel._required_slot_margin_bias` gains a schema-position gate
(`_required_slot_margin_position_accepts_slot`), reusing
`_schema_role_slot_bias`'s own `accepts_slot` convention exactly: on a
`component` frame, the active argument's schema must carry
`x-openui-placeholder` (the content properties — `label`, `text`, `title`,
`body`, ...); on an `object` frame, the active property's schema must be
able to reach a visible slot at all (`_schema_can_reach_visible_slot`). Any
other frame kind (`variadic`, `fixed`, ...) stays permissive — the traced
failure never touched those. Default (`0.0`) and the E617 contract-gated
guard are unchanged. `model.twotower` v61 → v62. New/updated unit tests:
`test_required_slot_margin_bias_excludes_non_content_schema_positions`
(new) and `test_required_slot_margin_bias_excludes_frame_depth_zero`
(updated so its frame_depth>=1 case uses a content-flagged schema, since it
was incidentally relying on missing-schema-info defaulting permissive).

## Re-verification: fixed, but not for free

**`rico_eval_test_25` is fully fixed.** Re-run at margin `{0,1,2,3,4}`:
every arm is now **byte-identical** to the margin=0 control —
`meaningful_program_v1_rate=1.0` in all 5 (was `1.0/0.0/0.0/0.0/0.0` in
E629). All 8 headline metrics match exactly across every margin. The 9
`required_slot_margin` fires still occur but are now confined to
`Button.label` (content) and permissive `variadic` array-item positions;
0 of 9 are `hijacked_non_slot_candidate`, and none change the final program.

**But the same fix reverts E626/E628's previously-reported `ood_dashboard_01`
gain, entirely.** Re-running E626/E628's own matched `n=4` OOD replay:

| Metric | margin=0 | margin=2 (post-fix) | margin=6 (post-fix) |
| --- | ---: | ---: | ---: |
| meaningful v1 | 0.5 | 0.5 | 0.5 |
| reward | 0.814 | 0.814 | 0.814 |
| fidelity | 0.55 | 0.55 | 0.55 |

Margins 2 and 6 are now byte-identical to control on all 4 OOD records —
`ood_dashboard_01` no longer improves. Reading why: pre-fix, margin=2's
"win" prediction was `v0 = Callout(":ood.dash.status.title",
":ood.dash.status.body", ":ood.dash.m2.value")` (3 *different* slots poured
into 3 positional args) and `v3 = Card([], ":ood.dash.m1.value",
":ood.dash.m1.value")` (empty children, 2 stuffed non-content args) — the
**identical over-stuffing mechanism** this fix removes, just landing
somewhere that happened to help rather than hurt. Decode is greedy/argmax,
so the entire remaining trajectory is sensitive to the first point of
divergence; once `required_slot_margin` no longer fires anywhere along this
path, the whole decode collapses back to control's, not just the one
stuffed argument. Separately: `Card([], ":ood.dash.m1.value", ...)` never
tripped `_is_meaningful_program`'s literal `"Card([])"` substring check
(which only matches an *exactly*-empty call, nothing after the array) even
though its children were just as empty as `rico_eval_test_25`'s rejected
`Card([])` — so part of the original Dashboard "gain" was itself an
artifact of that check missing a padded-but-still-empty Card, not genuine
new content. `ood_gallery_01`/`ood_modal_01`/`ood_auth_01` are unaffected.

**A widened n=19 re-sweep with the fix confirms this is not a wash — it's a
net loss on the binary gate.** Pooled across `held_out`/`rico_held`/
`adversarial`/`ood`/`smoke` (n=19, same union E629 used):

| Metric | margin=0 | margin∈{1,2} (post-fix) | margin∈{1,2} (E629, pre-fix, for reference) |
| --- | ---: | ---: | ---: |
| meaningful_program_v1 rate | 0.6842 (13/19) | **0.6316 (12/19)** | 0.6842 (13/19) |
| reward_score (mean) | 0.7889 | 0.7149 | 0.9260 |
| placeholder_fidelity (mean) | 0.5781 | 0.5825 | 0.8991 |
| structural_similarity (mean) | 0.4494 | 0.4221 | 0.5295 |
| component_type_recall (mean) | 0.5965 | 0.5570 | 0.6667 |

Exactly one record differs between margin=0 and margin=2 post-fix (of 19) —
and it is **neither** `rico_eval_test_25` (fixed, matches control now) nor
`ood_dashboard_01` (reverted, matches control now). It is `smoke_hero_01`,
newly regressing (`True -> False`, `empty_root_stack`): margin=0's own
natural prediction here is already a near-incoherent program (placeholders
and refs mixed directly as `Stack` children, a raw nested-list literal
inside the array) that nonetheless passes the coarse `_is_meaningful_program`
checks; at margin>=1 post-fix, the greedy decode instead collapses to
`Stack([])`. This was invisible pre-fix because the old over-stuffing
behavior happened to steer this already-chaotic path away from an empty
result — a third, independent instance of the same theme: on this
particular 800-step scratch checkpoint, greedy decode is fragile enough on
several of these 19 records that *any* decode-time bias touching the path
can flip the binary gate either way, largely disconnected from whether the
specific mechanism is "correct."

## Verdict

The `rico_eval_test_25` failure mode is genuinely root-caused and genuinely
fixed — concrete before/after trace evidence, byte-identical safety across
5 margins. The fix is principled (mirrors an existing, reviewed convention)
and keeps the default off, so it is kept. But the honest headline is not
"fixed, gains preserved" the way E628 was: re-verifying against both the
regressing record *and* the previously-confirmed gains (as this task
required) shows the gain was not separable from the bug — it was the same
mechanism — and a wider resweep exposes a third failure this checkpoint's
noise had been hiding. **`required_slot_margin_decode_weight`'s apparent net
benefit on this single small/noisy scratch checkpoint does not survive
closing the over-stuffing gaming path.** Default stays `0.0`. This is not a
confirmatory result in either direction.

## What this is not

A single scratch checkpoint (800 steps, seed 0); n=1 (trace) / n=4 (OOD
replay) / n=19 (widened suite, still below `DEFAULT_MIN_SUITE_N=20` per
suite). Not a ship claim. No checkpoint trained, promoted, or synced.
Margins 3/4 were not re-swept on the widened suite post-fix (E629 already
showed they add nothing over 1/2 pre-fix, and the net effect is already
negative at margin=1 post-fix).

## Next step (deferred)

1. `smoke_hero_01`'s regression was not independently traced — but given
   three different records now attributed to three different
   attributions (a real fix, a reverted gaming artifact, a newly-exposed
   fragility) on the *same* single checkpoint/seed, a fourth single-record
   trace is likely lower value than a genuinely powered multi-seed/retrained
   comparison before drawing any further conclusion about this lever.
2. E620's original coverage-aware component/property closure recommendation
   remains the next lever this lineage has not directly tried.
3. `_is_meaningful_program`'s literal `"Card([])"` / `"Stack([])"` substring
   checks miss an empty-children component padded with trailing non-content
   properties — observed evidence here, not fixed (different harness,
   `evals.meaningful_program`, out of this iteration's scope), worth
   hardening to a structural emptiness check in a future iteration.

Raw evidence:
[JSON](iter-e630-required-slot-margin-schema-position-gate-20260720.json).
