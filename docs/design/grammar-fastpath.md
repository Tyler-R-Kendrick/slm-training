# Grammar fast-path (force-emit + MaskGIT admit)

> **Goal law:** this document is bound by [decode-invariants.md](decode-invariants.md) —
> constrained decoding is the product, deterministic singleton bypass outranks
> learned scores, and a rejected approach never closes a goal.

## Goal

Skip transformer steps whenever exact grammar and scope authority has one legal
tokenizer continuation, including request-local semantic symbols, and reject
fills that make the completion language empty.

## Research lineage

| Idea | Citation | How we use it |
| --- | --- | --- |
| MaskGIT iterative unmask | Chang et al., CVPR 2022 · [arXiv:2202.04200](https://arxiv.org/abs/2202.04200) | Denoiser canvas + parallel unmask |
| Constrained diffusion / hole admit | Mündler et al., 2025 · [arXiv:2508.10111](https://arxiv.org/abs/2508.10111) · [constrained-diffusion.ai](https://constrained-diffusion.ai/) | **Adapted**: `admit_fill` checks left-span + hole completion instead of full CFG∩NFA emptiness |
| Speculative / forced structural emit | Leviathan et al., ICML 2023 · [arXiv:2211.17192](https://arxiv.org/abs/2211.17192) (adjacent; no draft LM) | **Adapted**: DFA singleton `=` `(` `)` `[` `]` `,` force-emit + `pick_constrained_token` |

Full fidelity tags and honesty rules: [research-lineage.md](research-lineage.md).

## Package

`slm_training.dsl.grammar.fastpath`:

| Module | Role |
| --- | --- |
| `engine.py` | `OpenUIIncrementalEngine` — Lex + feed tokens; `accepts()`; `is_deterministic_next()` |
| `completion_artifact.py` | Safe on-disk static LALR/token projection plus independent certified loader |
| `completion_kernel.py` | Request-local packed parser/semantic states, outgoing edges, bounded witnesses, and forced closures |
| `semantic_state.py` | Immutable scope, binder, schema, slot, and literal-frame facts updated per DSL-native token |
| `force_emit.py` | Map singleton terminal → tokenizer id; draft windows |
| `maskgit_constrain.py` | `admit_fill` — hole probe via benign `hole` substitution |
| `losses.py` | Cheap `force_align_loss` on gold `= ( ) [ ] ,` positions |
| `gate.py` | Optional sigmoid trust head (does not override DFA) |

The surface DFA force hint remains limited to narrow terminals:
`=` `(` `)` `[` `]` `,`. DSL-native semantic tokens may also bypass inference,
but only when the active pack returns a complete, scope-fingerprinted singleton
for the current request-local symbol table. A broad `NAME` / `COMPONENT` /
`STRING` terminal is never sufficient by itself.

Exact authority is ordered before every learned preference. A constrained decoder
asks the active DSL pack for a finite, scope-fingerprinted completion domain using
the current prefix, request-local runtime symbols, tokenizer projection, and its
remaining-token budget. The preferred domain carries replayable terminal
witnesses. If that bounded horizon proof is unavailable at an unscoped framed
literal boundary, decoding may use the pack's still-complete exact next-action
frontier; it never widens to the tokenizer vocabulary, and final document
certification remains mandatory. Request-local runtime-symbol domains continue
to fail closed rather than use that unscoped fallback. Logits rank the exact
domain only. The shared decoder contains no OpenUI component,
binder, schema, or renderer policy. A missing/incomplete next-action authority
or empty domain is a constrained dead end — never a
`keep_and_rank`/full-vocabulary fallback. Renderer visibility remains an independent
post-generation product policy, not a grammar rule.

## Decode wiring

- **LTR / repair** (`TwoTowerModel._constrained_ltr_repair`, `_greedy_ltr_decode_batch`):
  call `exact_forced_token_id` before the denoiser. For DSL-native tokenizers it
  asks the pack-owned completion domain first, so a scope-aware semantic
  singleton skips the forward even when there is no structural force hint.
  Other tokenizers still distinguish a significant-lexeme hint from a full
  tokenizer-token singleton. Only exact authority skips the forward. Batched LTR
  compacts the remaining ambiguous rows; repair commits exact decisions without
  fabricating model logits or log-probabilities. Legal whitespace keeps the
  compositional path model-ranked when it can change source bytes.
- **MaskGIT** (`_generate_maskgit_one`): when exactly one canvas position remains
  unknown and no model-dependent remask follows, a strict DFA singleton that also
  passes the active admit/stream checks commits before the denoiser. All multi-hole,
  remask-active, incomplete-proof, or rejected cases retain the ordinary neural
  proposal. In `mask`/`hybrid` mode, candidate fills still run `admit_fill`; leave a
  position masked on reject. Grammar-on picks never commit DFA-illegal tokens.
- **Compiler tree** (`_compiler_ltr_decode_one`): obtains the same pack-owned
  domain directly. Exact complete singletons commit without a neural call.
  Ambiguous future trie states are materialized as speculative canvases and
  sent through bounded batches; forced single-child states never enter those
  batches. Zero-valued `compiler_prefill_max_states` and
  `compiler_prefill_token_budget` select a device-aware bound.
- **ONNX serving** (`OnnxTwoTowerModel`): ranks the same budgeted domain and
  checks for an exact singleton before invoking ONNX Runtime. An unavailable
  proof produces a tagged certified fallback, never EOS substitution or an
  unconstrained vocabulary token.
- **Certify** (`_ensure_valid_openui`): every public document-generation path
  validates, repairs if configured, and otherwise emits a tagged minimal valid
  fallback. `grammar_constrained=False`, stochastic grammar selection, and
  unconstrained fallback flags are rejected at public/model-build boundaries;
  legacy checkpoint metadata is normalized to the mandatory safe policy.
- **Train aux**: `fastpath_aux_weight` (CLI `--fastpath-aux-weight`) adds
  `force_align_loss` without walking the DFA every step.

## Packed incremental completion kernel

Lexer-native OpenUI completion is owned by one request-local
`CompletionSession`. Its integer state handles intern pairs of LALR control
state and immutable `SemanticState`; the handle never enters certificates or
serialized output. Direct DSL-native token feeds advance both halves without
re-lexing the rendered prefix. Parser forks copy control stacks only, while
semantic authority remains in the independent persistent state.

The semantic state replaces repeated prefix scans with Python-integer masks and
interned facts for declarations, references, dependency reachability, active
calls/arrays, slot use, string/literal frames, and schema requirements. The
scan-based helpers remain executable differential references, not a production
owner.

For an interned state `s` and remaining room `r`, bounded witness search follows
the existing forest edge order with strictly decreasing token cost. Each
top-level candidate retains the historical 16-node allowance and 64-entry
query-local LRU, preserving the outward `CompletionDomainV1` status, candidate
order, witnesses, terminals, and reasons. A partial forest or depleted search
is typed `UNKNOWN`; it is never cached as `UNSUPPORTED`. Only replayable
positive paths become witnesses. This also preserves the established difficult
prefix contract: `root = Card([b1,` exposes exactly 12 replayable ordered
candidates in both the production kernel and V1 differential reference. Its
outward V1 status is `complete` with reason `witness_pruned`: the bounded
reference search leaves the additional `Form` and `Table` branches unknown,
so exact parity narrows to the 12 proven siblings; it does not prove those two
unknown branches unreachable.

`forced_closure(s, r)` follows only complete single-edge forests and stops at an
ambiguity, EOS, literal boundary, incomplete authority, or the token horizon.
Compiler decode consumes that closure before neural ranking. Decode rows retain
their current state handle across commits and rollback, and equivalent rows may
share immutable completed domains while keeping logits row-local. The verified
support solver likewise advances packed state handles; proof payloads remain
token/digest based.

Greedy compiler batches synchronize independent rows at those completion
checkpoints. Complete singletons and forced closures commit before tensor
packing; only ambiguous row canvases enter a shared denoiser forward, bounded
by the existing device-aware prefill limits. Parser/semantic state, contracts,
runtime symbols, biases, and final validation remain per row. Speculative
ranking and rollback/trajectory search retain the sequential decoder because
their control state can choose or restore a row before tensor work is known.

All prefix/state/domain caches are lifetime-bounded by the request, decode row,
or batch and occupy O(unique packed states/prefixes); they have no
process-global arbitrary-prefix forest cache. The request-independent LALR
control relation and tokenizer-to-terminal projection are compiled into the
checked artifact described in
[certified-completion-artifact-and-tps-target.md](certified-completion-artifact-and-tps-target.md)
and the claim-family split in
[adr-constrained-diffusion-topology-split.md](adr-constrained-diffusion-topology-split.md).
The decoder consumes only the projection after exact live equivalence passes;
scope and semantic state are never serialized. E1 production snapshots live in
`static_control_domain.py`; warm ~89.8× hard-prefix is `request_local_memo_reuse`
only. `TimeoutError` is checked before
cache reuse and after forest/closure construction and propagates through
parser, witness, and solver loops. Native OpenUI production uses the packed
path. The prefix-oriented reference remains a differential oracle and the
compatibility path for tokenizers without lexer-native kind ids (including
current word-tokenizer ONNX exports).

OpenUI remains deterministic LALR: its ambiguous work is semantic candidate
ranking, not generalized parsing. A GSS/SPPF dependency would add runtime and
packaging cost without changing this grammar's deterministic control path.

## Offline compiler contract

The fast path must make the same decisions in a clean worktree that has no
OpenUI bridge `node_modules`. The official component schema is therefore
committed as `dsl/grammars/openui_schema.json`; `lang_core.library_schema()`
prefers the live pinned bridge and falls back to that snapshot. Run
`python -m scripts.sync_openui_schema --check` wherever bridge dependencies are
installed to prove exact schema and property-order parity. Property order is
part of the positional language contract and the snapshot must not be sorted.

The in-process Lark grammar is also authoritative for offline AST completion.
Statements require a newline separator, and prefix decoding preserves a final
newline even though user-facing final decoding trims it. This keeps partial
lexical state distinct from final-document formatting. Compiler-tree admission
is derived from the grammar and schema; it does not inspect parser error strings
or match known output literals. This correction changes the language contract
ID from `f2d0c69ba5849ef9` to `dffa3760e8008c2c`. The separator helper is
grammar-hidden so generated AST consumers continue to receive statements
directly beneath `start`.

## Cactus

Header sketches live under `src/slm_training/runtime/cactus/kernels/` (not compiled).
PyTorch remains the train/eval path; export via `cactus.export_checkpoint_bundle`.

## Config

`TwoTowerConfig.grammar_fastpath` (default True), `grammar_fastpath_mode`
(`force` | `mask` | `hybrid`), `fastpath_aux_weight`.

Compiler-tree ranking remains opt-in; constrained generation does not. A compiler singleton is exact only when its
completion forest reports `coverage="complete"`; a partial singleton still runs
the neural ranker and is not counted as a certified forced span. MaskGIT's narrow
one-hole terminal step can bypass; every step whose schedule, confidence, attention,
survival, or remasking could depend on logits remains neural work.
