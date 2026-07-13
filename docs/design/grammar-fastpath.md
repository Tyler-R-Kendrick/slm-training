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

`slm_training.grammar_fastpath`:

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

## Cactus

Header sketches live under `src/slm_training/cactus/kernels/` (not compiled).
PyTorch remains the train/eval path; export via `cactus.export_checkpoint_bundle`.

## Config

`TwoTowerConfig.grammar_fastpath` (default True), `grammar_fastpath_mode`
(`force` | `mask` | `hybrid`), `fastpath_aux_weight`.
