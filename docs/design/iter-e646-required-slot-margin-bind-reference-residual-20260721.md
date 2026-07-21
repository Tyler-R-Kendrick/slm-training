# E646 — the residual bind-reference "hijack" is not a bug (negative result, plus an additive trace field)

Date: 2026-07-21
Status: trace complete; no bias/gate behavior change; additive instrumentation
only; not a ship claim

## Which open item this picks

E645 fixed the variadic-frame schema-position gate and left one item flagged
as unfixed: post-fix, `hijacked_non_slot_candidate=true` fires still occur at
positions whose item schema *is* slot-reachable (so E645's gate correctly
leaves them active), but where the pre-bias argmax was a bind-reference token
(`kind='bind'`, e.g. `&2`) rather than a component-opening token. Counts at
margin=2: `held_out` 1/14, `rico_held` 4/16, `smoke` 1/7, `adversarial` 0/6,
`ood` 0/12. This session traces the largest of those (`held_out`, `rico_held`)
plus `smoke` directly to find out whether it is a real quality problem or
benign/desirable behavior.

## Environment note

Unlike E645 (whose target checkpoint did not survive across sessions), this
session found E645's own checkpoint still present:
`outputs/runs/e645-smoke-hero-trace-scratch800-20260721/checkpoints/last.pt`,
sha256 `a4c24987d1cf3dca97f84fd19e64cfc94210de27b75dc27cc4f7eb5f2041b7e7` —
byte-identical to E645's recorded hash. No retraining was needed; all numbers
below are directly comparable to E645's own numbers, not a new mechanism
match on a different checkpoint.

## Part 1: the residual counts reproduce exactly

Re-running E645's exact post-fix reverification command (`held_out` n=5,
`rico_held` n=3, `required_slot_margin_decode_weight=2`) and reading
`decode_stats.constrained_selection_traces` directly reproduced E645's counts
precisely: `held_out` 1/14, `rico_held` 4/16 (2 unique fire sites, each
recorded twice — the eval pipeline decodes each record twice internally).
Same checkpoint, same mechanism, no drift.

## Part 2: E645's own characterization was partly inaccurate

E645's `residual_hijacks_post_fix` note stated *all* residual hijacks were
"at variadic-frame positions ... where the pre-bias argmax happened to be a
bind-reference token." Reading the `held_out` fire's trace directly falsifies
the "all": its `frame_path` is `[{kind: component, expr_type: element:Input,
arg_index: 1}]` — a `component` frame, not `variadic` — and `before_kind` is
`"struct"` (`before_token: "-"`, the negation operator), not `"bind"`. Only
the `rico_held` fires (both on `rico_eval_test_25`) match E645's description:
`frame_path` `[Card(component,arg0), variadic(array,arg0),
TextContent(component,arg0)]`, `before_kind='bind'` (`&2`) →
`chosen_kind='sym'` (`@1`/`@7`).

Also worth correcting: the *active* frame at fire time (`frames[-1]`) for the
`rico_held` cases is the innermost `component` frame (`TextContent`'s own
content argument, already gated by **E630's** `x-openui-placeholder` check),
not the `variadic` frame itself. E645's own variadic-frame gate
(`frame.schemas[0]` reachability) isn't what makes these positions eligible
— E630's pre-existing component-frame gate is. E645's narrative for this
residual class was right about the bind-vs-sym token swap but wrong about
which frame kind and gate governs it. This is a correction to E645's written
record, not a new bug — noted because this lineage's docs get read forward as
ground truth by later sessions.

## Part 3: direct trace reads of 4 concrete instances

For each hijack site, this session ran the *same* checkpoint at
`required_slot_margin_decode_weight=0` (bias off — the pre-bias/pre-hijack
path is what actually gets emitted) and `=2` (bias on), and diffed the full
predictions plus `placeholder_fidelity`/`structural_similarity`, rather than
inferring correctness from `top_candidates` scores alone.

**`rico_eval_test_25` (`rico_held`)** — `Card([Tabs([TabItem(...)])])`'s
2nd/3rd positional args:

```
margin=0: v4 = Card([Tabs([TabItem(":cardview_1.title", ":cardview.body", ":cardview.title")])])
margin=2: v4 = Card([Tabs([TabItem(":cardview_1.title", ":card4.body", ":card4.title")])])
```

At margin=0 (bias off, bind-reference wins naturally) the decoder *reuses*
already-emitted placeholder text (`:cardview.body`, `:cardview.title`,
duplicated from `v1`/`v2` earlier in the same program) instead of the
distinct, still-unused gold slots (`:card4.body`, `:card4.title`). At
margin=2 the bias floors the still-missing required slot above the
bind-reference and picks the correct distinct placeholders instead.
`placeholder_fidelity`: **0.625 → 0.875**. `structural_similarity` unchanged
(0.6546). `meaningful_program_v1` `True` at both margins.

**`smoke_callout_01` (`smoke`)** — `Callout(...)`'s 3rd positional arg:

```
margin=0: root = Callout("itet", ":smoke.callout.title", ":smoke.callout.title")
margin=2: root = Callout("itet", ":smoke.callout.title", ":smoke.callout.heading")
```

Same pattern: margin=0 duplicates `:smoke.callout.title`; margin=2 selects
the distinct still-missing `:smoke.callout.heading` slot instead.
`placeholder_fidelity`: **0.333 → 0.667**. `meaningful_program_v1` `True` at
both margins.

**`smoke_hero_01` (`smoke`)** — exhibits the same bind→sym swap mechanism at
a nested `variadic > TextContent(arg0)` position, but `placeholder_fidelity`
is already `1.0` at both margins — a true no-op, neither helped nor hurt.

**`held_out_form_01` (`held_out`)** — the `struct`/`-` fire from Part 2 (not
the bind-reference class). Both margins' predictions on this record are
already fairly degenerate on this undertrained 800-step checkpoint (margin=0
nests `Button` inside `Button` inside `Button`; margin=2 produces a long flat
array of repeated placeholder items) — messier evidence than the other three,
but directionally consistent, not a counter-example: `structural_similarity`
**0.4333 → 0.6**, `placeholder_fidelity` unchanged (0.8 → 0.8), no
regression.

## Part 4: verdict — no bias/gate fix

Across all 4 traced instances, overriding the displaced candidate with the
still-missing required slot never lowered `placeholder_fidelity` or
`structural_similarity`, and twice materially *raised* `placeholder_fidelity`
by replacing stale/duplicate placeholder reuse with the correct distinct gold
slot. This is the **opposite** of E628/E630/E645's traced mechanism, where
the displaced candidate was a real component-opening token the bias had no
business displacing. On this evidence, the residual
`hijacked_non_slot_candidate=true` fires are quality-neutral or
quality-improving, not a bug. Per this repo's values (never force a fix that
isn't warranted, never weaken a gate to pass a number), **no change was made
to `_required_slot_margin_bias` or
`_required_slot_margin_position_accepts_slot`.**

Caveat: 4 instances on 1 scratch checkpoint (800 steps, seed 0) is not a
powered sample; "never lowered fidelity" is an observation on this small
evidence set, not a proof it can never happen elsewhere. The `held_out`
struct-token instance is the least clean of the four (an already-degenerate
record) and is reported as such, not smoothed over.

## Part 5: one additive, behavior-neutral change

`_record_required_slot_margin_trace` (`src/slm_training/models/twotower.py`)
now also emits `hijacked_bind_reference_candidate: bool` — true iff the
displaced pre-bias argmax candidate's kind was `"bind"`, distinguishing it
from the broader `hijacked_non_slot_candidate` (which also fires for
`component`/`struct`/`builtin`/`lit` displacements — the genuinely dangerous
class E628/E630/E645 fixed). This is **purely additive**: no bias/gate
function changed, no decode-time token selection changed, no existing trace
field's value changed. A future session mining these traces for genuine
structural hijacks can now filter `hijacked_non_slot_candidate and not
hijacked_bind_reference_candidate` instead of re-deriving this session's
finding from scratch — matching this lineage's own pattern (E627/E640) of
building better observability rather than leaving a finding only in prose.

`model.twotower` v81 → v82 (instrumentation-only; no-op for decode
behavior). New test:
`tests/test_models/test_compiler_decode.py::test_required_slot_margin_trace_distinguishes_bind_reference_hijacks`.
148/148 passed across `test_compiler_decode.py` + `test_choice_tokenizer.py`
+ `test_decode_stats.py` (147 from E645 + 1 new).
`python -m scripts.verify_version_stamps --check`: ok after the
`versions.json` bump.

## Honest verdict

Negative/neutral result on the substance (no bias/gate behavior change
warranted), plus a correction to E645's own written characterization of this
residual class, plus one additive, tested, version-bumped instrumentation
field that operationalizes the finding for future sessions. 4 traced
instances on 1 scratch checkpoint (800 steps, seed 0); not a powered sample;
not a ship claim; no checkpoint trained, promoted, or synced (E645's
checkpoint reused verbatim, confirmed byte-identical by sha256); default
`required_slot_margin_decode_weight` (`0.0`) unchanged; E617 gating
unchanged.

## Next step (deferred)

1. E620's coverage-aware component/property closure recommendation remains
   the next untried lever this lineage has repeatedly deferred (E643, E644,
   E645, now E646).
2. A genuinely powered multi-seed/retrained comparison of
   `required_slot_margin_decode_weight` remains open (E626's own deferral).
3. `hijacked_bind_reference_candidate` (added here) now lets a future session
   filter directly for the genuinely-dangerous component/struct/builtin-kind
   hijacks across a wider suite, without re-deriving this session's
   bind-reference finding.
4. The `held_out_form_01` struct-token (`-`) displacement (Part 2/3) was
   traced only shallowly (the messiest of the 4 instances, on an
   already-degenerate record) — worth a closer look if this position class
   recurs on a better-trained checkpoint.

Raw evidence:
[JSON](iter-e646-required-slot-margin-bind-reference-residual-20260721.json).
