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
| `models/onnx_inference.py` · `_forced_singleton` | same bypass in the ONNX serving backend |

**Status: implemented, all backends.** ONNX was the last gap — it forwarded
first and consulted force-emit afterwards; it now proves the singleton before
the session runs.

### I2 — Forced bypass on singletons

When the scope-aware symbol table (DFA domain / `CompletionDomainV1` /
choice-codec state) shows **exactly one** valid next symbol, that symbol is
committed with **no neural forward and no ranking**, in every decode path and
every backend. A partial proof refuses to bypass — `exact_forced_token_id`
returns `None` unless the DFA-allowed set is exactly `{forced}`, and the ONNX
`_forced_singleton` additionally requires `coverage == "complete"`.

Certainty is never downgraded into a soft preference.

**A new decode path ships with a bypass test or it does not merge.** The
canonical shape is the `forwards_count == 0` assertion in
`tests/test_models/test_inference_speed.py`; the ONNX equivalent is
`tests/test_web/test_onnx_inference.py::test_onnx_forced_tokens_cost_zero_denoiser_runs`.
`scripts/verify_decode_invariants.py` fails CI when a registered decode backend
has no such test.

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

**Status: machinery shipped, default `off`.** Turning it on for serving needs a
preregistered `ExperimentCampaignV1` with the table's `corpus_fingerprint`
bound to the run. The technique itself is a lever: n-gram ↔ trie ↔ learned
ranker may be swapped by campaign, as long as verification gates every commit.

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

**Where the checkpoint input comes from.** The batched LTR loop passes
`forced_run_lengths=None`: at plan time it has no proof about positions *after*
`t`, because a forced run beyond the current position only becomes provable
once `t` is committed — and the next iteration then discovers it for free via
the ordinary bypass. So in that loop the planner contributes minimal-row
compaction plus device-window sizing, and `schedule_checkpoint_hits` stays 0.
Callers that already hold a forced-run draft (`draft_forced_ids`) pass one and
get boundary truncation; that is the seam for extending checkpoint scheduling
to the repair and ONNX paths.

**Status: planner shipped, default `off`** (`prefill_schedule`,
`prefill_schedule_max_lookahead`). With the lever off the planner reproduces
the caller's legacy budget exactly, so it is observational until a campaign
turns it on. Utilization regressions are measured from those counters, never
asserted.

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
| `models/twotower.py` · `allow_unconstrained_fallback` | **default `False`**; the unfiltered retry is opt-in and diagnostic |
| `web/service.py` | absolute contract — raises `GenerationExhausted` rather than handing the UI an invalid constrained sample; forces `allow_unconstrained_fallback=False` on the serving config |
| `models/onnx_inference.py` | raises `GrammarCertificationError` instead of returning uncertified text |
| `harnesses/model_build/eval_policy.py` | every named strict policy is checked by `require_constrained_production_config` before the run |

An empty legal domain is a constrained dead end, never a full-vocabulary
fallback.

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
`--no-grammar` still resolves to it), the HTTP `grammar_constrained=false`
field — whose attempts are stamped `diagnostic_control: true` and whose result
reports `certified=false` even when the text happens to parse — and the
`current_native` decode-path spec used by `harnesses/eval/ablate_decode_scaffolding.py`.

**Status: enforced.** The former default-on unconstrained retry, the
uncertified ONNX return, and the unlabeled HTTP control arm are all closed.

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

**Status: open goal.** A reserved op-token channel exists decoder-side and
default-off (`dsl/operators/reserved_tokens.py`) with fail-closed checkpoint
compatibility. The towers still use different vocabularies
([`dsl-native-tokenizer.md`](dsl-native-tokenizer.md) frames the interface as
deliberately asymmetric), and nothing ops-like exists encoder-side.

**Rejected approach, live goal.** e803 measured and rejected
*decoder-target* op tokens on this corpus
([`e803-reserved-operator-baseline-20260723/summary.md`](e803-reserved-operator-baseline-20260723/summary.md)).
That experiment says nothing about *encoder-side* ops sharing, which is what
I13 actually requires. **Successor approach:** define `OPS_VOCAB v1` as a
reserved, content-addressed op-token set shared by both towers, and preregister
an encoder-conditioned campaign before training. Until such an experiment
falsifies it, the invariant stands.

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

**Status: partial — full-AST output is the shipped default.** The reserved
patch-target arm was experimentally rejected, and the SLM-299/305 reachability
audit measured `reachable_fraction = 0.0` from the standard seed on all suites
([`iter-slm305-edit-language-20260724.md`](iter-slm305-edit-language-20260724.md)).

**Rejected approach, live goal.** Full-AST output is the *bootstrap* mode, not
the end state. **Successor approach:** attack reachability first —
reachability-aware seed selection, macro actions, reachability-certified
training pairs (the SLM-299 analyzer already exists) — before any retrial of
patch-as-default-target.

---

## IV. Goal-drift guard

### I14 — Goals are non-negotiable; approaches are disposable

A rejected experiment closes an *approach*, never a *goal*. Every rejected
approach to an invariant above files its successor approach — or an explicit,
dated, documented waiver — in the same measured-results doc, and links it here.

Status labels like `rejected`, `unavailable`, `nl_available=False`, and
`reachable_fraction=0.0` describe **current approach state**. They may never be
cited as a reason an invariant does not apply.

Open goals with named successors, at a glance:

| Invariant | Rejected approach | Successor approach |
| --- | --- | --- |
| I13 | e803 decoder-target op tokens | shared `OPS_VOCAB v1`, encoder-conditioned campaign |
| I12 | patch-as-default-target (SLM-299/305) | reachability-aware seeds / macro actions / certified pairs |
| I11 | — (never attempted) | CRDT-converging merge, replacing conflict rejection |
| I10 | — (rung unbuilt) | simplified-NL inventory as the bridge to complex NL |
| I3 | — (machinery new) | certify the n-gram ranker by campaign, then default-on for serving |
| I4 | — (machinery new) | certify checkpoint scheduling by campaign, then default-on for serving |

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
