# Decode invariants (goal law)

This repository is **not training a natural-language LLM.** It trains a
**grammar-constrained symbolic diffusion model** that emits templated grammars —
scaffolded structure plus structural reasoning. Templated / natural-language
content is deferred to a separate real LLM (`OUTPUT_CONTRACT_VERSION = 2`,
`src/slm_training/dsl/language_contract.py`).

Everything below is a **goal invariant**. It constrains every model, harness,
lever, experiment, doc, and agent action here — past, present, and future. The
canonical agent-facing statement lives in
[`AGENTS.md` § Non-negotiable architecture invariants](../../AGENTS.md); this
document is its expansion: what each invariant means, where it is implemented,
what its current status is, and — where an approach was rejected — which
successor approach carries the goal forward.

Machine enforcement: `python -m scripts.verify_decode_invariants` (CI, static;
no torch required). Component: `decode.invariants` in
`src/slm_training/resources/versions.json`.

Papers this design cites: constrained decoding of diffusion LLMs with CFGs
(arXiv:2508.10111, "IG-CD"), and Lookahead-then-Verify (arXiv:2602.00612,
"LAVE").

---

## I. Constrained decoding is the product

### I1 — Deterministic completion paths bypass inference

Where a deterministic answer exists, no forward pass runs. Authoritative
deterministic decode proofs always outrank learned, semantic, confidence, or
preference scores.

| Where | What |
| --- | --- |
| `dsl/grammar/fastpath/engine.py` · `is_deterministic_next` | DFA proves the next lexeme is fully determined |
| `dsl/grammar/fastpath/force_emit.py` · `force_next_token_id` / `draft_forced_ids` | forced token id, and forced multi-token drafts |
| `models/grammar.py` · `exact_forced_token_id` | strengthens the proof to *sole legal tokenizer token* before bypassing |
| `models/twotower.py` LTR / batched / MaskGIT loops | commit forced rows with the row removed from the forward |
| `models/onnx_inference.py` | derives the completion forest before every forward; a singleton domain commits with no denoiser run |

**Status: implemented, all backends.** ONNX was the last structural gap — it
forwarded first and consulted force-emit afterwards; it now proves the
singleton before the session runs. Short-horizon repair used to discard an
otherwise-proven bypass; that is closed too (see I2).

### I2 — Forced bypass on singletons

When the scope-aware symbol table (DFA domain / `CompletionDomainV1` /
choice-codec state) shows **exactly one** valid next symbol, that symbol is
committed with **no neural forward and no ranking**, in every decode path and
every backend. A partial proof refuses to bypass — `exact_forced_token_id`
fails closed unless the exact authorities prove one continuation, and the ONNX
loop requires `coverage == "complete"` before it will read the domain at all.

Certainty is never downgraded into a soft preference.

**A new decode path ships with a bypass test or it does not merge.** The
canonical shape is the `forwards_count == 0` assertion in
`tests/test_models/test_inference_speed.py`; the ONNX equivalent is
`tests/test_web/test_onnx_inference.py::test_onnx_forced_tokens_cost_zero_denoiser_runs`.
`scripts/verify_decode_invariants.py` fails CI when a registered decode backend
has no such test. That gate is static — it proves the assertion exists; pytest
proves it holds.

#### Horizon-limited domains are not contradictions

`exact_forced_token_id` proves a singleton two ways. For the DSL-native codec
it reads the pack's completion domain, which is scope-aware and therefore
catches semantic singletons the DFA cannot see. That domain needs budget: with
a short `remaining_tokens` it cannot enumerate a terminal witness and returns
`coverage="none"`.

`"none"` means *nothing was proven*, not *the singleton was refuted*, so the
DFA proof is consulted instead of throwing the decision away. A **complete**
domain naming more than one candidate does refute a singleton, and still
refuses. The whitespace veto in the DFA proof is skipped for the native codec,
because the native completion domain already excludes insignificant whitespace
from its candidate set — applying it only in the fallback would make the same
position disagree with itself depending on remaining budget. Whitespace is not
a symbol under the symbol-only contract, and emitting the forced lexeme in its
place cannot make a program illegal.

This closed `test_repair_exact_token_skips_forward_and_records_authority`,
which was red on `main` from before this document existed.

### I3 — Speculative completion from forward-calculated symbol tables

Symbol tables are computed *before* the model. At a non-singleton branch point
the choice among already-legal candidates may be made by a **deterministic
scorer**, and a run of such choices committed as one span — lookahead-then-
verify (LAVE), with the grammar oracle as the verifier (IG-CD
intersection-witness completions).

Implementation: `src/slm_training/dsl/grammar/fastpath/speculative_rank.py`.

- `NgramTableV1` — back-off n-gram over decoder token ids, content-addressed,
  built train-split-only by `scripts/build_speculative_ngram_table.py`.
- `SpeculativeRankerV1.choose` — total, reproducible order over the *legal*
  domain; commits without a forward only when the top-vs-runner-up margin
  clears `speculative_rank_margin`.
- `speculative_span` — drafts a verified continuation: each step re-derives the
  completion domain, so a partial proof, an empty domain, or a close call stops
  the span rather than guessing.

Seam: `TwoTowerModel._select_compiler_path` consults the ranker after the
singleton shortcut and **before** `_denoiser_hidden`, so a confident decision
genuinely skips inference. Levers: `speculative_rank` (`off` | `ngram`),
`speculative_rank_table`, `speculative_rank_margin`. Counters:
`speculative_rank_evaluations` / `_commits` / `_tokens` / `_declined` in
`DecodeStats`.

**Two rules hold without exception.** Ranking is a lever; legality is not —
nothing in that module can add a candidate. And speculation verifies against
the grammar oracle before it commits.

**The committed table.** `src/slm_training/resources/decode/speculative_ngram_v1.json`
is built train-split-only from the immutable certified corpus
(`openui_verified_v1`, 1682 records, 89,415 native tokens, order 3, 523
contexts). Targets are templatized first, so the table is keyed on symbols and
placeholders and never on free-form string content. Setting
`speculative_rank="ngram"` without naming a table resolves to it, so the lever
is reachable with no build step. `scripts/build_speculative_ngram_table.py
--check` fails when the artifact and its builder disagree.

It ranks real branch points confidently: after `root = ` (27 legal candidates)
it picks `Stack(` at margin 1.0, and after `root = Stack([` (25 candidates) it
picks `<BIND_1>` at margin 1.59 — both decided from the symbol table with no
forward.

**Status: machinery shipped and reachable, default `off`.** Turning it on for
serving needs a preregistered `ExperimentCampaignV1` binding the table's
`corpus_fingerprint` to the run. The technique itself is a lever: n-gram ↔ trie
↔ learned ranker may be swapped by campaign, as long as verification gates
every commit.

### I4 — Symbol tables schedule compute

The symbol table is not only a legality filter, it is a schedule. Before every
denoiser call the grammar already says which rows are still ambiguous, where
the next **grammar checkpoint** is (the first position at which the domain
collapses to a singleton and the model's opinion stops mattering), and
therefore how much canvas is worth reading.

Implementation: `src/slm_training/runtime/decode_schedule.py`
(`plan_prefill`, `next_grammar_checkpoint`, `record_plan`). The planner is
pure — it returns a `PrefillPlanV1` and never runs a forward.

- Rows the deterministic bypass already resolved are dropped from the forward
  in **every** mode; that follows from I2, not from the lever.
- `mode="checkpoint"` sizes the window by detected device
  (`runtime/accel.detect_device`) — accelerator backends never get a window
  narrower than the legacy one, because a launch is the expensive part there —
  and truncates it at the next grammar checkpoint when the caller supplies a
  forced-run draft.
- Counters: `scheduled_prefills`, `scheduled_rows_skipped`,
  `scheduled_prefill_tokens_saved`, `schedule_checkpoint_hits`.

**Where the checkpoint input comes from.** `common_forced_run` answers the one
thing the grammar can prove about positions beyond `t` before `t` is decided:
if *every* legal candidate at `t` leads into the same length of forced lexemes,
those positions are determined no matter what the model picks. That makes
`t + 1` a real checkpoint, and the forward only needs to reach it.

The probe walks the DFA force-emit oracle (far cheaper than a completion
forest) from each candidate, and is bounded on both axes — at most 8 candidates
and 4 lexemes deep, over at most 4 batch rows, taking the minimum across all of
them. Exceeding any budget returns 0, which claims nothing and leaves the
planner on device-window sizing. `TwoTowerModel._grammar_checkpoints` runs it
only when the lever is on, so the default path pays nothing.

**Status: planner shipped and fed by the symbol table, default `off`**
(`prefill_schedule`, `prefill_schedule_max_lookahead`). With the lever off the
planner reproduces the caller's legacy budget exactly and the probe never runs,
so it is observational until a campaign turns it on. Utilization regressions
are measured from those counters, never asserted.

### I5 — The speculative technique may evolve

n-grams, tries, learned rankers, successor caches (E74
`models/speculative_denoise.py`) are all admissible **provided I6 holds**.
Levers are registered in `src/slm_training/levers.py` and preregistered as
campaigns; see [`experiment-campaign-governance.md`](experiment-campaign-governance.md).

### I6 — Never emit invalid grammar

Every production decode path is grammar-constrained end to end. Unconstrained
arms are **diagnostic controls only** — never production defaults, never
serving paths, and their output is never certified, shipped, or gated on.

Fail-closed points:

| Surface | Behavior |
| --- | --- |
| `models/grammar.py` · `pick_constrained_token` | refuses rather than emitting unconstrained top-1 when legality cannot be certified |
| `models/grammar.py` · `require_constrained_generation` | an unconstrained request is refused outright, not honored quietly |
| `models/twotower.py` · `allow_unconstrained_fallback` | **default `False`**, and the MaskGIT unconstrained retry is gone |
| `harnesses/model_build/eval_policy.py` · `MANDATORY_GENERATION_POLICY` | a floor under *every* policy, checkpoint-declared included; `require_constrained_production_config` checks it before the run |
| `web/service.py` | absolute contract — raises `GenerationExhausted` rather than handing the UI an invalid constrained sample; forces `allow_unconstrained_fallback=False` on the serving config |
| `models/onnx_inference.py` | substitutes a certified deterministic program rather than returning uncertified text, and reports `fallback_used` |

An empty legal domain is a constrained dead end, never a full-vocabulary
fallback.

**A certified substitute is not a successful decode.** ONNX satisfies I6 by
returning a certified deterministic program when its own decode cannot be
certified. That program parses, so a caller that only checks "does it parse"
would record a failed decode as a real model attempt — and the playground
persists every attempt as annotation evidence. `web/service.py` therefore reads
`consume_generation_evidence()` after each attempt and raises
`SubstitutedGeneration` on `fallback_used`, so the substitute is counted as the
failure it is. Any new backend with a substitution path must expose the same
evidence flag.

**Registered weakening levers.** `levers.CONSTRAINT_WEAKENING_LEVERS` names
every lever that can make output less constrained or spend a forward where a
proof existed, with its fail-closed value:

| Lever | Safe value | Invariant | Effect |
| --- | --- | --- | --- |
| `grammar_constrained` | `True` | I6 | legality |
| `allow_unconstrained_fallback` | `False` | I6 | legality |
| `grammar_fastpath` | `True` | I2 | bypass |

They surface in the lever catalog as `weakens_constraint: true` /
`diagnostic_only: true`, and `require_constrained_production_config` blocks
them from production and ship-gated configurations.

Named diagnostic controls (allowed, clearly labeled, never shipped):
`--unconstrained-control` on `scripts/train_model.py` (the deprecated spelling
`--no-grammar` still resolves to it), and the HTTP `grammar_constrained=false`
field — whose attempts are stamped `diagnostic_control: true` and whose result
reports `certified=false` even when the text happens to parse. Note that
`current_native` is **no longer** such a control: it now inherits
`MANDATORY_GENERATION_POLICY` and is constrained end to end.

**Status: enforced.** The former default-on unconstrained retry, the
uncertified ONNX return, the MaskGIT unconstrained fallback, and the
substituted-decode-as-success gap on the serving path are all closed.

---

## II. What the model is (and is not)

### I9 — Output is scaffolded grammar

Model targets contain only grammar/AST literals and placeholder symbols
(`dsl/language_contract.py`, `assert_symbol_only_output`, `dsl/placeholders.py`).
Surface realization is late (`dsl/surface.py`,
`models/surface_autoregressor.py`); literal copy is deferred to a separate
model ([`openui-twotower.md`](openui-twotower.md)).

Natural-language vocabulary is **optional fluff** — optimizable later, never
load-bearing, never a ship blocker, removable without breaking the contract.

**Status: implemented.** See
[`symbol-only-output-contract.md`](symbol-only-output-contract.md).

### I10 — The use-case ladder

In order, no skipping. Each rung is certified before the next opens; the
`CERT_CAP*` fail-closed gates stay.

| Rung | Status | Evidence |
| --- | --- | --- |
| AST → AST | built | tree-edit diffusion (`models/tree_edit_diffusion.py`) + edit corpora |
| grammar → AST | built | prompt→AST corpus |
| grammar+ops → AST | built | `harnesses/train_data/operator_corpus.py` |
| simplified-NL → AST | thin | frozen frontier families L3–L5, tiny inventory |
| complex NL → AST | **unbuilt** | fail-closed `nl_available=False`, `CERT_CAP1_unavailable` |

The gate ordering is correct — the top rung is *unbuilt*, not *abandoned*. Its
successor approach is the simplified-NL inventory as a bridge; current position
and blockers are restated in [`../MODEL_CARD.md`](../MODEL_CARD.md).

### I10b — Calculator/solver enhanced with inference

Not a chat model. Inference fills ambiguity; it never authors structure a
deterministic solver can derive. This is the same ordering as I1, stated as a
product claim rather than a decode rule.

---

## III. Encoder/decoder vocabulary and multi-turn

### I13 — Reserved compute-ops vocabulary, shared encoder ↔ decoder

The encoder vocabulary MUST reserve a compute-ops vocabulary — AST / graph /
set / topology operations — **known and shared by the decoder vocabulary** (one
versioned `OPS_VOCAB`, content-addressed, both towers). Grammar symbols layer
on top of that shared ops base. NL vocabulary sits above grammar symbols and is
strictly optional.

Implementation: `src/slm_training/dsl/ops_vocab.py`, reserved in the versioned
`ops` token-id namespace (`0x10000`–`0x11000`, above every existing codec
range, so no stored embedding row moves).

Two properties make this a vocabulary rather than one more list to keep in sync:

- **Derived, not authored.** Every entry comes from a live operator registry
  (`operators.local`, `operators.topology`, `operators.conversation`). An op
  cannot be in the model's vocabulary without an implementation, and cannot be
  implemented without appearing here — `build_ops_vocab` raises on either.
- **One mapping, not two conventions.** `shared_token_ids()` is the only source
  of op token ids. "Shared encoder↔decoder" is not something both towers are
  asked to honor; it is the same function called twice, and
  `assert_shared_across_towers` rejects a tower that is missing an op or
  renumbers one.

| Family | Ops |
| --- | --- |
| `ast` (node-local value edits) | `replace_node`, `set_property`, `unset_property` |
| `graph` (edge rewiring) | `move_node`, `reparent_node`, `duplicate_subtree` |
| `set` (child membership/order) | `add_child`, `remove_node`, `reorder_children` |
| `topology` (subtree shape) | `wrap_node`, `unwrap_node`, `expand_template`, `contract_subtree` |
| `history` (the event store itself) | `ast_edit`, `undo`, `redo`, `checkout_state`, `fork`, `copy_state` |

19 ops, content-addressed, pinned in
`src/slm_training/resources/ops_vocab_registry.json` and gated by
`verify_decode_invariants`. `assert_layering` proves grammar symbols sit in
their own namespaces above the reserved base and never inside it. Natural
language is absent by construction — that is the point: it is optional fluff
above the grammar layer and must never become load-bearing.

**Status: vocabulary reserved and shared; tower wiring is the open rung.** The
contract, the ids, the drift gate, and the layering proof exist. What remains
is conditioning the context tower on these tokens and training against them —
a preregistered campaign, distinct from e803, which measured and rejected
*decoder-target* op tokens
([`e803-reserved-operator-baseline-20260723/summary.md`](e803-reserved-operator-baseline-20260723/summary.md))
and said nothing about encoder-side sharing. The decoder-side reserved-token
channel (`dsl/operators/reserved_tokens.py`) stays default-off until then.

### I11 — Multi-turn is a CRDT event store

A conversation is an append-only, content-addressed event store of operations
on the conversation AST (`dsl/operators/conversation.py`,
`ConversationTraceV1`). Turn inputs are constrained to ops on that AST. Replay
is exact and cursor-contiguous — no hidden state
(`replay_conversation_trace`).

Operations: `AST_EDIT`, `UNDO`, `REDO`, `CHECKOUT_STATE`, `FORK`, and
`COPY_STATE`. `FORK` opens a new branch; `COPY_STATE` re-materializes an
earlier state's artifact on the *current* branch, and refuses a cross-branch
source because reference tables are branch-scoped.

**Status: partial — event store yes, CRDT no.**
`dsl/operators/merge.py` is a conservative three-way merge that *rejects*
conflicts rather than converging. That is a **documented interim state, not the
goal**. **Successor approach:** CRDT semantics for the merge — commutative ops
or last-writer-wins registers per node-attribute — so concurrent branches
converge without operator intervention. Tracked as an open goal here; a
conflict-rejecting merge may never be cited as evidence the invariant does not
apply.

### I12 — Patch/diff outputs across turns

Turns emit operation patches/diffs, not full rewrites, wherever the edit space
can reach the target. The AST artifact is a **materialization of the entire
conversation history** — full replay, no hidden cursors.

The contract is in place: an 11-action tree-edit language
(`models/tree_edit_diffusion.py`, SLM-305) and `TurnArtifactV1` carrying
`OperatorApplicationV1` deltas.

**Status: partial — full-AST output is the shipped default for the measured
variant.** The reserved patch-target arm was experimentally rejected. The
`reachable_fraction = 0.0` finding is scoped to the **`tree_edit_diffusion`**
variant only (`action_alphabet_id="tree_edit_diffusion.edit_actions"`,
`action_alphabet_fingerprint` `ab2662a497d8359ffaee46ebbd4bee3789f5b0f2accaf8bf46c5dee489622dab`
as of SLM-305's 11-action alphabet —
`slm_training.dsl.variants.build_variant_contracts()`), measured from the
standard seed on all suites
([`iter-slm305-edit-language-20260724.md`](iter-slm305-edit-language-20260724.md)).
The **`repl_operators`** and **`twotower_prompt_ast`** variants are
`NOT_MEASURED` for this invariant (see [I14](#i14--goals-are-non-negotiable-approaches-are-disposable)'s
scoping rule below). VAR0-02 publishes the full `(variant, suite)` matrix in
[`var0-02-reachability-matrix-20260726.md`](var0-02-reachability-matrix-20260726.md):
`repl_operators` is `not_measured_deferred` and `twotower_prompt_ast` is
`not_applicable` — reachability is a `(variant, suite)` cell, never a
program-wide scalar.

**Rejected approach, live goal.** Full-AST output is the *bootstrap* mode for
`tree_edit_diffusion`, not the end state. **Successor approach, ordered by
the measured reason histogram** (`iter-slm305-edit-language-20260724.md`
recorded exactly two reason codes across every suite: `unsupported_component`
and `needs_direction_change` — no other reason code was observed):

1. **Property-mutation action class** (maps to `needs_direction_change`,
   e.g. `train_button_row_01`, `rico_eval_test_0`). VAR1-01's hypothetical
   probe
   ([`var1-01-set-property-probe-20260725.md`](var1-01-set-property-probe-20260725.md))
   confirmed one genuine `PROVEN_REACHABLE` flip via a what-if `set_property`
   action on a case previously blocked by `needs_direction_change`
   (`adv_empty_prompt_01`), licensing VAR1-02 to add a real `SET_PROPERTY`
   action to `TreeEditSpace`.
2. **Pack-derived component inventory** (maps to `unsupported_component`,
   e.g. `train_auth_01`, `held_out_form_01`). These cases need a wider
   component/property inventory sourced from the pack, not a new action
   class — a distinct successor from (1).
3. **Seed selection and macro actions** — demoted below (1) and (2). Per
   `slm299_edit_reachability.py:347-354`, the root's `rest` must match the
   seed's in every mode because the root is never removed or re-minted, so a
   different seed changes *which* targets are reachable but cannot
   substitute for a missing action class on root-owned properties.
   Reachability-certified training pairs (the SLM-299 analyzer already
   exists) remain a valid follow-on once (1) and (2) close the action-class
   and inventory gaps.

---

## IV. Goal-drift guard

### I14 — Goals are non-negotiable; approaches are disposable

A rejected experiment closes an *approach*, never a *goal*. Every rejected
approach to an invariant above files its successor approach — or an explicit,
dated, documented waiver — in the same measured-results doc, and links it here.

Status labels like `rejected`, `unavailable`, `nl_available=False`, and
`reachable_fraction=0.0` describe **current approach state**. They may never be
cited as a reason an invariant does not apply. A variant-scoped measurement
may also never be cited as a program-scoped status (VAR0-02): a
`reachable_fraction` is attributed to one registered variant's action
alphabet and says nothing about any other variant until that variant is
separately measured.

Open goals with named successors, at a glance:

| Invariant | Rejected approach | Successor approach |
| --- | --- | --- |
| I13 | e803 decoder-target op tokens | `OPS_VOCAB v1` now reserved + shared; next is an encoder-conditioned campaign |
| I12 | patch-as-default-target for `tree_edit_diffusion` (SLM-299/305) | property-mutation action class (VAR1-01/02) → pack-derived inventory (VAR0-03) → seeds/macro actions/certified pairs, demoted |
| I11 | — (never attempted) | CRDT-converging merge, replacing conflict rejection |
| I10 | — (rung unbuilt) | simplified-NL inventory as the bridge to complex NL |
| I3 | — (machinery new) | certify the committed n-gram table by campaign, then default-on for serving |
| I4 | — (machinery new) | certify checkpoint-aligned prefills by campaign, then default-on for serving |

### I14a — Variant-scoped measurements are not program-scoped statuses

SLM-427 (VAR1-03) established this scoping rule after I12's original text
cited a single variant's `reachable_fraction = 0.0` as if it applied to the
whole program, and named a successor (seed selection) that did not follow
from the measured reason codes (`unsupported_component`,
`needs_direction_change` — both properties of the action alphabet and
inventory, not of seeds or training pairs).

* A measurement taken against one variant's action alphabet
  (`VariantContractV1.variant_id` /
  `action_alphabet_fingerprint`) may be cited only for that variant. Every
  other registered variant is `NOT_MEASURED` for the same invariant until
  independently run — it is never assumed to inherit the result.
* A successor approach listed under an invariant must be traceable to at
  least one reason code actually present in that measurement's published
  histogram. A successor that does not map to any observed reason code is
  not a valid successor, however plausible it sounds.

### I7 — Every agent surface carries the law

The repo configures several coding harnesses, each reading a different
instruction file. An invariant stated only in `AGENTS.md` reaches only the
agents that happen to read it, so every surface must carry it:
[`../../AGENTS.md`](../../AGENTS.md), [`../../CLAUDE.md`](../../CLAUDE.md),
[`../../GEMINI.md`](../../GEMINI.md),
[`../../.github/copilot-instructions.md`](../../.github/copilot-instructions.md),
[`../../.cursor/rules/decode-invariants.mdc`](../../.cursor/rules/decode-invariants.mdc),
[`../../.grok/workflows/autotrain.rhai`](../../.grok/workflows/autotrain.rhai),
and the skills that run experiments (`autotrain`, `honest-ship-eval`,
`improve-openui-harnesses`, `running-experiment-matrices`).

`python -m scripts.verify_agent_surfaces` owns the obligation × surface matrix
and is authoritative — `verify_decode_invariants` delegates to it for the
`decode.invariants` obligation rather than keeping a second copy. It certifies
the *other* repository laws (run cap, iron law, honest gates, data-quality
loop, model card, version stamps, dashboard parity, preregistered campaigns) on
the same surfaces; each of those had drifted off at least one surface before it
existed. It also certifies hook parity, so the post-edit checks cannot stay one
harness's privilege. Background:
[`agent-harness-parity-audit.md`](agent-harness-parity-audit.md).

Surfaces cite the `I*` ids used here. Renumbering them locally makes
"invariant 11" mean different things in different files, which defeats the
point of a canonical statement.

### I15 — Everything is documented

These invariants live canonically in this file. They are linked from
[`../../README.md`](../../README.md), [`../MODEL_CARD.md`](../MODEL_CARD.md),
[`grammar-fastpath.md`](grammar-fastpath.md),
[`symbol-only-output-contract.md`](symbol-only-output-contract.md),
[`dsl-native-tokenizer.md`](dsl-native-tokenizer.md),
[`speculative-denoising.md`](speculative-denoising.md), and
[`dsh3-08-conversation-state-graph-20260723.md`](dsh3-08-conversation-state-graph-20260723.md),
and they are carried into OpenWiki regeneration via
[`../openwiki/INSTRUCTIONS.md`](../openwiki/INSTRUCTIONS.md).

Changing an invariant means editing this file, bumping the `decode.invariants`
component in `src/slm_training/resources/versions.json`, and passing
`python -m scripts.verify_decode_invariants` in CI. A silent weakening is a
regression and blocks merge.

---

## What this intentionally permits

So agents do not over-correct:

- Swapping speculative techniques (n-gram ↔ trie ↔ learned ranker) — allowed,
  via preregistered campaign, as long as grammar verification gates every commit.
- New decode levers — allowed, if registered, version-stamped, and non-weakening.
- Diagnostic unconstrained control arms in eval harnesses — allowed, clearly
  named, never shipped.
- Deferring NL surface polish — allowed and encouraged; NL is fluff by design.
