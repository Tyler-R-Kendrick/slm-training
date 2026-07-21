# E631 — fixing `meaningful_program_v1`'s empty-children detection gap (and the honest re-scoring it demands)

Date: 2026-07-21
Status: metric bug verified against real code/data, fixed conservatively
(purely additive), regression-tested, and re-scored against the real E626
checkpoint's actual predictions -- not a ship claim; not a confirmatory
result about `required_slot_margin_decode_weight` in either direction.

## Which deferral this picks

E630 root-caused `rico_eval_test_25`'s over-stuffing and fixed it in
`TwoTowerModel._required_slot_margin_bias`, but along the way it found (and
explicitly deferred, calling it "a different harness, out of scope") that
`_is_meaningful_program`'s literal `"Card([])"` substring check misses an
empty-children `Card` whose remaining positional arguments got padded with
unrelated stuffed values -- and that this gap was itself responsible for part
of E626/E628's originally-reported `ood_dashboard_01` gain. This iteration
picks up that exact deferral.

## Verifying the diagnosis against real code and real data first

E630's report is a paraphrase, not a spec, so before touching anything: read
`_is_meaningful_program` in full
(`src/slm_training/harnesses/model_build/eval_runner.py`), and parsed the
exact padded strings E629/E630 reported through the real DSL parser
(`slm_training.dsl.parser.validate`) to see the actual AST.

The literal checks are:

```python
if "Stack([])" in compact or "Stack([]," in compact:
    return False, "empty_root_stack", serialized
if "Card([])" in compact:
    return False, "empty_card", serialized
```

This confirms E630's diagnosis, with one sharper detail E630 didn't call out:
**Stack's own check already has a `"Stack([],"` clause for the padded case**
-- it was only `Card` that lacked the equivalent companion clause. The gap
was a real, narrow asymmetry already visible in the code, not merely an
abstract "one syntactic shape."

Parsing `Card([], ":a", ":b")` through the real parser shows:

```json
{"type": "element", "typeName": "Card",
 "props": {"children": [], "variant": ":a", "direction": ":b"}}
```

`children` lives in `props` as its own key, structurally separate from the
other (stuffed) prop values -- exactly the semantic signal needed. A quick
schema check
(`src/slm_training/dsl/grammars/openui_schema.json`) turned up two more
component types the old literal check never covered *at all*: `Modal` and
`Carousel` both declare a required `children` array, and
`root = Modal(":t", true, [])` / `root = Carousel([])` both passed
`meaningful_program_v1` pre-fix.

## The fix

A new `_first_empty_children_component(node)` helper walks the real parsed
AST (`Program.root`, not the serialized text) looking for any element whose
`props["children"]` is an empty list, returning that node's `typeName`. It is
called from `_is_meaningful_program` **immediately after, not instead of**,
the two existing literal checks:

```python
if "Stack([])" in compact or "Stack([]," in compact:
    return False, "empty_root_stack", serialized
if "Card([])" in compact:
    return False, "empty_card", serialized
empty_type = _first_empty_children_component(program.root)
if empty_type == "Stack":
    return False, "empty_root_stack", serialized
if empty_type == "Card":
    return False, "empty_card", serialized
if empty_type is not None:
    return False, f"empty_children:{empty_type}", serialized
```

**Why this cannot weaken the gate.** The literal branches are untouched and
still run first with their original reason strings; the AST check only adds
new rejections the literal checks miss (it is reached only when neither
literal branch already returned `False`). There is no code path where the
AST check causes something the literal checks used to reject to now pass.
Stack/Card keep their original reason strings when the AST check is what
actually fires on them (nested cases, or Card's padded-args case) --
`empty_root_stack` / `empty_card` -- so no existing consumer of those two
reason strings sees new vocabulary. Only genuinely new types (Modal,
Carousel) get a new reason code, `empty_children:<TypeName>`.
`decode_feasibility.py::classify_parse_failure` buckets that new reason
alongside the existing two under the `trivial_layout` diagnostic bucket
(display-only, does not affect the pass/fail outcome).
`harness.model_build.eval` bumped v33 -> v34.

## Tests (`tests/test_evals/test_meaningful_program.py`)

New parametrized cases directly reproduce the exact shapes from E629/E630:

- `Card([], ":stuffed.variant")` -> `empty_card` (the `rico_eval_test_25` shape)
- `Card([], ":ood.dash.m1.value", ":ood.dash.m1.value")` -> `empty_card` (the
  `ood_dashboard_01` shape)
- A nested `Stack([Card(non-empty), Card([], stuffed)])` -> `empty_card`
  (nesting doesn't evade it)
- `Modal(":title", true, [])` -> `empty_children:Modal` (new coverage)
- `Carousel([])` -> `empty_children:Carousel` (new coverage)

Regression guards for cases that must still pass:

- `Card([TextContent(":a")], ":real.variant")` -> `True` (non-empty content
  plus other props present)
- `Stack([Card([TextContent(":a")])], "column")` -> `True` (nested non-empty)

The pre-existing `test_meaningful_program_v1_backward_lock` parametrization
(`Stack([])`/`Card([])`/populated Stack/`Separator()`) is untouched and still
passes exactly as before -- the literal-check cases are bit-for-bit
preserved. 36/36 passed in this file (29 pre-existing + 7 new); 56/56 passed
across `test_audit_meaningful_program.py`, `test_matrix_meaningful_metrics.py`,
`test_decode_invariance.py`, `test_eval_gates.py`, `test_language_contract.py`.

(A pre-existing, unrelated failure in
`tests/test_harness_core/test_gate_engine_golden.py` -- a stale golden
fixture expecting `binding_aware_meaningful_v2` version `"2.0.0"` when the
registry already has `"2.1.0"` from a prior session's `no-bump:` note --
reproduces identically on a clean checkout before this session's changes;
not touched here, out of this iteration's scope.)

## Honest re-scoring against the real E626 checkpoint

Re-running the metric fix in isolation isn't enough -- the task required
checking whether any *previously-passing* record's verdict flips under the
stricter check, using the real checkpoint E626-E630 all reused (sha256
`c5b7c807…dd561221`, verified again this session; no retrain, no further
decode-time code change -- E630's `twotower.py` fix stays exactly as
committed).

**Pooled n=19 (`held_out`+`rico_held`+`adversarial`+`ood`+`smoke`), margin=0
control (the lever fully inactive):**

| | before this session's metric fix | after |
| --- | ---: | ---: |
| `meaningful_program_v1_rate` | 0.6842 (13/19) | **0.5789 (11/19)** |

The entire drop is `rico_held`: 1.0 (3/3) -> 0.3333 (1/3). Monkeypatching
`_is_meaningful_program` to print `(record_id, reason, prediction)` while
running the real `evaluate_model` CLI (not inferring from aggregates) shows
exactly which records and why:

- `rico_eval_test_25`: unaffected, stays `True` -- no empty-children node in
  its margin=0 prediction.
- `rico_eval_test_42`: **flips `True -> False`, reason `empty_card`.**
  Prediction includes `v5 = Card([], ":card2.title")`,
  `v6 = Card([], ":card3.body")`, `v9 = Card([], ":card2.body")` -- three
  `Card([], <stuffed value>)` nodes, present in the checkpoint's *unmodified
  baseline* decode (margin=0, lever fully off). This over-stuffing shape is
  not specific to `required_slot_margin_decode_weight` at all -- it is a
  pre-existing artifact of this particular 800-step scratch checkpoint's own
  decode, invisible to the old literal check regardless of margin.
- `rico_eval_test_77`: **flips `True -> False`, reason `empty_card`,** same
  shape (`Card([], ":coupon_container_2.title")` etc.), same reasoning.

**Pooled n=19, margin=2 (post-E630-decode-fix):**

| | before this session's metric fix (E630) | after |
| --- | ---: | ---: |
| `meaningful_program_v1_rate` | 0.6316 (12/19) | **0.5263 (10/19)** |

Per suite after the metric fix: `held_out` 0.6 (unchanged binary rate, one
record's continuous metrics shift -- a cosmetic margin-sensitive swap that
doesn't cross the gate, the same kind of diff E629 already documented
elsewhere); `rico_held` 0.3333 (byte-identical predictions/metrics between
margin=0 and margin=2 -- `rico_eval_test_25` stays fixed per E630,
`rico_eval_test_42`/`77` are equally rejected at both margins since their
padding is unrelated to the lever); `adversarial` 0.5 (byte-identical,
matches E630); `ood` 0.5 (byte-identical, confirms E630's own finding that
`ood_dashboard_01`'s gain is fully reverted, and the metric fix adds nothing
further here -- no Modal/Carousel in this suite's predictions); `smoke`
drops 1.0 -> 0.6667, `smoke_hero_01`'s already-documented (E630) regression,
driven by the decode bias itself, unaffected by this session's metric change.

**Net conclusion under the corrected metric.** Both `rico_eval_test_42` and
`rico_eval_test_77` fail identically at every margin (they're unrelated to
the lever), so they shift the absolute baseline down at both margin=0 and
margin=2 by the same two records and cancel out of the delta that actually
matters. The margin=2-vs-margin=0 gap is unchanged in sign: -1 record either
way (12→13 pre-metric-fix per E630, now 10→11 post-metric-fix) -- still net
non-positive. **The metric fix does not change E630's headline verdict about
`required_slot_margin_decode_weight`** (fixed regression, reverted gain, net
non-positive on this checkpoint); it only corrects the absolute baseline
level by catching two real, pre-existing, lever-independent scoring
artifacts that had nothing to do with the experiment under test.

## Verdict

E630's diagnosis holds up against real code and real data, with one added
precision (Stack already had the padded-args clause Card lacked). The fix is
purely additive -- verified with tests for the new catch, the two
previously-caught cases (unchanged reason strings), and non-empty
pass-through cases -- and cannot loosen any existing rejection. Re-scoring
the real checkpoint's actual predictions confirms two previously
(incorrectly) passing records now correctly fail, at every margin including
pure control: an expected, honest tightening, not a regression to hide. Not
a ship claim; no checkpoint trained, promoted, or synced.

## What this is not

A single scratch checkpoint (800 steps, seed 0); n=19 (widened suite, below
`DEFAULT_MIN_SUITE_N=20` per suite). Does not retrain or resweep margins 3/4
(already shown to add nothing over 1/2). Does not touch
`evals/meaningful_program.py` (`binding_aware_meaningful_v2` / strict v2) --
confirmed by direct inspection to have no literal empty-substring check of
this kind, so nothing there needed fixing.

## Next step (deferred)

1. E620's coverage-aware component/property closure recommendation remains
   the next untried lever.
2. A genuinely powered multi-seed/retrained comparison of
   `required_slot_margin_decode_weight` remains open (E626's own deferral,
   still not picked up through E631).
3. `smoke_hero_01`'s regression (first found in E630, unaffected by this
   session's metric fix) remains untraced.

Raw evidence:
[JSON](iter-e631-meaningful-program-v1-empty-children-ast-fix-20260721.json).
