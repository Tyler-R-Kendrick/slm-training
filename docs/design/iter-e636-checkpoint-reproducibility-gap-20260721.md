# E636 — checkpoint reproducibility gap blocks the Modal follow-up

Date: 2026-07-21
Status: blocked; not reproducible; no treatment run; not ship

E635 named the next step explicitly: fix Modal's optional enum-valued `size`
argument (observed as a garbage byte-spelled literal, e.g. `"itet"`, appended
after `title`/`open`/`children`) using the same active-property compatibility
principle that fixed Auth, without disturbing the now-correct Auth path. The
existing `schema_enum_close_decode_weight` lever (already wired in
`TwoTowerModel._schema_enum_close_bias` and `evaluate_model.py`, currently 0.0
in every E630-E635 recipe) is the natural candidate: E593 previously showed
it removes optional-enum garbage on a different, older checkpoint at the cost
of a Dashboard regression, so re-testing it on the current lineage is a
legitimate next ablation.

## What blocked it

E630-E635 all say they "reused" E620's rejected local-only scratch checkpoint
(`e620-required-slot-coverage-scratch800-20260720`,
SHA-256 `3ce5c9efc70ed69c7a6680129018ec0aa2061f56020ff42301faf144363ecc5f`).
That checkpoint lives only under `outputs/`, which is gitignored and, per
`docs/design/checkpoint-bucket.md`, scratch/CI runs are `--no-sync-checkpoints`
(local-only by design). This session started from a clean checkout with no
`outputs/` directory, so the checkpoint referenced by five prior experiments
did not exist. The reuse contract silently depended on the same working
directory persisting across the sessions that produced E630-E635; it does not
survive a fresh checkout.

Two retraining attempts reproduced E620's exact documented recipe (train dir
`e530_visible_semantic_roles_r2_20260719`, `data_manifest_sha
e65a6ac5a7c49499b638582c325eafd0b245cc4aa9d2650d1396a88230eccee2`, 244
records, `twotower`, scratch context, choice tokenizer, device cpu, 800 steps,
batch size 1, seed 0):

| Attempt | Training flags | Loss | Checkpoint SHA-256 |
| --- | --- | ---: | --- |
| Historical E620 | (documented fields only) | 4.068010 | `3ce5c9ef…` |
| v1 (this session) | matches documented fields | 4.068013 | `8ab4f5de…` |
| v2 (this session) | v1 + `--schema-in-context --slot-contract-in-context --honest-slot-contract --slot-contract-constrained-decode` | 3.992113 | `d29a8175…` |

The data manifest SHA and record count match exactly and the loss is within
noise, so the recipe itself reproduces. Neither attempt reproduces the
checkpoint bytes, and — more importantly — neither reproduces the qualitative
decode behavior on the exact E635 OOD `n=4` eval recipe (`suite ood`,
`context-backend scratch`, `honest-slot-contract`,
`slot-contract-constrained-decode`, `semantic-role-contract-in-context`,
`schema-opaque-decode-weight 4.0`, `slot-coverage-close-decode-weight 2.0`,
`grammar-ltr-max-tokens 160`):

| OOD `n=4` | E635 r2 (reference) | v1 (`e636-baseline-r1`) | v2 (`e636-baseline-v2-r1`) |
| --- | ---: | ---: | ---: |
| meaningful v1 / strict v2 | 0.7500 / 0.2500 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |
| fidelity / validity | 0.6750 / 0.8050 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| structure / component recall | 0.5729 / 0.6250 | 0.1627 / 0.1458 | 0.1291 / 0.2083 |
| reward | 0.8515 | 0.7058 | 0.9490 |
| AST node / edge F1 | 0.6357 / 0.5125 | 0.1000 / 0.0000 | 0.2833 / 0.0000 |
| AgentV | 0/1 | 0/1 | 0/1 |

Both fresh checkpoints produce degenerate unresolved-variable soup on all
four OOD records (`root = v0`, dozens of unused `v1..v17` bindings) and never
reach a literal `Modal(...)`, `Input(...)`, or `Button(...)` constructor call.
An 800-step scratch TwoTower is barely converged, and greedy argmax decoding
over ~150+ tokens is chaotic under those conditions: the tiny cross-environment
floating-point differences from an unpinned `torch` build (this session
resolved `torch==2.5.1` fresh; the original run's exact build isn't recorded)
are enough to flip early argmax choices and cascade into a completely
different generation path, even though training loss and data lineage match.

## Decision

Do not run the `schema_enum_close_decode_weight` treatment against either
substitute checkpoint: since neither ever emits a `Modal(...)` call, the
specific `size`-argument mismatch this experiment targets cannot occur in
their decode paths, so any comparison would be fabricated evidence, not a
real test of the hypothesis. No checkpoint was synced or promoted; both `v1`
and `v2` are discarded local scratch artifacts. No ship claim.

**Recommendation for the next session that picks up the Modal follow-up:**
either (1) sync/persist a designated frozen checkpoint for this active
multi-session ablation lineage instead of `--no-sync-checkpoints`, (2) pin
exact `torch`/`numpy` build versions so 800-step scratch trains are
byte-reproducible across sessions, or (3) migrate this ablation lineage to a
fully-trained/promoted checkpoint that isn't this sensitive to environment
noise. Until one of those lands, the E630-E635 checkpoint-reuse thread is not
continuable from a fresh checkout.

Evidence: [JSON](iter-e636-checkpoint-reproducibility-gap-20260721.json).
