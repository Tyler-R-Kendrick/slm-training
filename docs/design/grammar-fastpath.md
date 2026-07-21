# Grammar fast-path (force-emit + MaskGIT admit)

## Goal

Skip transformer steps when the OpenUI LALR acceptor has a **singleton structural
continuation**, and reject MaskGIT fills that make the CFG completion language empty.

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
| `force_emit.py` | Map singleton terminal → tokenizer id; draft windows |
| `maskgit_constrain.py` | `admit_fill` — hole probe via benign `hole` substitution |
| `losses.py` | Cheap `force_align_loss` on gold `= ( ) [ ] ,` positions |
| `gate.py` | Optional sigmoid trust head (does not override DFA) |

Force only narrow terminals: `=` `(` `)` `[` `]` `,`. Never force `NAME` /
`COMPONENT` / `STRING`.

Exact authority is ordered before every learned preference. Once the DFA, choice
state, complete compiler forest, or another authoritative decoder proves one legal
continuation, semantic bias, confidence, plan scoring, and model logits may not
replace or rerank it. Incomplete proofs remain ambiguous and fail closed to the
ordinary legal model-ranked path.

## Decode wiring

- **LTR / repair** (`TwoTowerModel._constrained_ltr_repair`, `_greedy_ltr_decode_batch`):
  call `force_emit_token_id` before the denoiser, then use
  `exact_forced_token_id` to distinguish a significant-lexeme hint from a full
  tokenizer-token singleton. Only the latter skips the forward. Batched LTR
  compacts the remaining ambiguous rows; repair commits exact decisions without
  fabricating model logits or log-probabilities. Legal whitespace keeps the
  compositional path model-ranked when it can change source bytes.
- **MaskGIT** (`_generate_maskgit_one`): when exactly one canvas position remains
  unknown and no model-dependent remask follows, a strict DFA singleton that also
  passes the active admit/stream checks commits before the denoiser. All multi-hole,
  remask-active, incomplete-proof, or rejected cases retain the ordinary neural
  proposal. In `mask`/`hybrid` mode, candidate fills still run `admit_fill`; leave a
  position masked on reject. Grammar-on picks never commit DFA-illegal tokens.
- **Certify** (`_ensure_valid_openui`): LTR repair → minimal valid fallback → raise
  when `grammar_finalize_validate` is set.
- **Train aux**: `fastpath_aux_weight` (CLI `--fastpath-aux-weight`) adds
  `force_align_loss` without walking the DFA every step.

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

Compiler completion remains opt-in. A compiler singleton is exact only when its
completion forest reports `coverage="complete"`; a partial singleton still runs
the neural ranker and is not counted as a certified forced span. MaskGIT's narrow
one-hole terminal step can bypass; every step whose schedule, confidence, attention,
survival, or remasking could depend on logits remains neural work.
