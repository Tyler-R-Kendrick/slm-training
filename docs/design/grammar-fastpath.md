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

## Decode wiring

- **LTR / repair** (`TwoTowerModel._constrained_ltr_repair`, `_greedy_ltr_decode_batch`):
  call `force_emit_token_id` before the denoiser; skip the forward when forced;
  always route the id through `pick_constrained_token` (DFA + stream probe).
- **MaskGIT** (`_generate_maskgit_one`): when `grammar_fastpath_mode` is `mask` or
  `hybrid`, run `admit_fill` on candidate fills; leave position masked on reject;
  grammar-on picks never commit DFA-illegal tokens.
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
