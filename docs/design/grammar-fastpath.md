# Grammar fast-path (force-emit + MaskGIT admit)

## Goal

Skip transformer steps when the OpenUI LALR acceptor has a **singleton structural
continuation**, and reject MaskGIT fills that make the CFG completion language empty.

Inspired by [constrained-diffusion.ai](https://constrained-diffusion.ai/) (CFG ∩
completion language emptiness checks), specialized to OpenUI via Lark
`InteractiveParser`.

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
  call `force_emit_token_id` before the denoiser; skip the forward when forced.
- **MaskGIT** (`_generate_maskgit_one`): when `grammar_fastpath_mode` is `mask` or
  `hybrid`, run `admit_fill` on candidate fills; leave position masked on reject.
- **Train aux**: `fastpath_aux_weight` (CLI `--fastpath-aux-weight`) adds
  `force_align_loss` without walking the DFA every step.

## Cactus

Header sketches live under `src/slm_training/cactus/kernels/` (not compiled).
PyTorch remains the train/eval path; export via `cactus.export_checkpoint_bundle`.

## Config

`TwoTowerConfig.grammar_fastpath` (default True), `grammar_fastpath_mode`
(`force` | `mask` | `hybrid`), `fastpath_aux_weight`.
