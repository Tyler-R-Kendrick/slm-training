# Is the 30s decode-timeout wall "almost enough" or "nowhere close"? — NOT SHIP

**Honesty:** `fixture_or_scratch`, isolated single-record probes (`--eval-limit
1`), 4 trials across 2 `held_out` records x 3 `--decode-timeout-seconds`
values. **Not ship. Diagnostic only** — the production default
`DEFAULT_DECODE_TIMEOUT_SECONDS` in `src/slm_training/levers.py` is untouched
and no ship gate changed.

## Task

Three prior levers against the same `held_out` baseline — decode-plumbing
micro-optimizations (PRs #1189-#1195, capstone re-measured in
[`lever-hard-decode-timeout-wall-heldout-capstone.md`](lever-hard-decode-timeout-wall-heldout-capstone.md)),
a 2.16x larger corpus + short SFT
([PR #1196 follow-up](lever-mix-loadable-v2-heldout-sft-vs-capstone-baseline.md)),
and chained-resume SFT to 256 steps
([PR #1197 follow-up](lever-chained-sft-resume-s256-heldout-vs-capstone-baseline.md))
— all left `meaningful_program_rate` at 0.0, with every `held_out` record
hitting `decode_outcome=runtime_timeout` at the 30s hard wall. This
iteration's question, verbatim from the assignment: **if a held_out record is
given more wall-clock budget, does its compiler-tree decode converge to a
real completion, or does it never terminate?** That distinction matters
architecturally (I3: constrained decoding is the product) — a miscalibrated
eval timeout is a different, much smaller problem than a decode algorithm
that is fundamentally unbounded on real input.

## Recipe

Reused PR #1198's 256-step chained-resume checkpoint unchanged from disk (no
rebuild needed): `outputs/runs/chain_lever_mix_v2_s256_lr1e3_bs2_sb15_seed47/checkpoints/last.pt`,
`checkpoint_sha256=9341ca71de0fd4a39a4936fabe5feb91fef43cf5d2167b501ec5fb6d487deb01`.
`code_commit=d278634` (this branch's parent HEAD).

Each trial is a **separate bounded Bash invocation**, isolated to one
`held_out` record via `--eval-limit 1 --eval-offset {0,1}`, holding everything
else fixed at the capstone recipe (`--grammar-constrained --seed 47
--constraint-debt-routing-mode fixed_asap --run-class scratch_matrix`):

```bash
python -m scripts.evaluate_model \
  --test-dir src/slm_training/resources/data/eval/e938_role_safe_all_targets_v2 \
  --suite held_out --eval-limit 1 --eval-offset {0,1} \
  --model twotower --device cpu \
  --checkpoint outputs/runs/chain_lever_mix_v2_s256_lr1e3_bs2_sb15_seed47/checkpoints/last.pt \
  --grammar-constrained --decode-timeout-seconds {30,60,90} --seed 47 \
  --constraint-debt-routing-mode fixed_asap --run-class scratch_matrix \
  --run-id diag_wall{30,60,90}_off{0,1}_s256_seed47
```

**Why single-record probes, not a full suite at a longer wall:** 5 records x
90s worst case = 450s, far over `MAX_RUN_MINUTES=3`'s 170s per-invocation
budget. One record at up to 90s fits comfortably (~103s observed wall-clock),
so isolation — not suite size — is what let this diagnostic run at all.

## Measured: 4/4 trials `runtime_timeout`, latency pinned exactly at whichever wall was configured

| trial | record | decode_timeout_seconds | command wall-clock | decode_outcome | latency_ms_p50 |
| --- | --- | ---: | ---: | --- | ---: |
| 1 | `held_out_form_01` (offset 0) | 30 | 38.5s | `runtime_timeout` | 30000.42 |
| 2 | `held_out_form_01` (offset 0) | 60 | 68.8s | `runtime_timeout` | 60100.48 |
| 3 | `held_out_form_01` (offset 0) | 90 | 103.3s | `runtime_timeout` | 90201.21 |
| 4 | `held_out_dual_card_01` (offset 1) | 60 | 68.9s | `runtime_timeout` | 60000.49 |

Every trial: `parse_rate=0.0`, `meaningful_program_rate=0.0`,
`failure_breakdown={parse_error: 1}`, `decode_timeout_count=1`. Full JSON:
[`lever-decode-timeout-wall-extension-single-record-probe.json`](lever-decode-timeout-wall-extension-single-record-probe.json).

## Interpretation: nowhere close, not almost enough

Tripling the wall on the *same* record (`held_out_form_01`, 30s -> 60s -> 90s)
produced the identical outcome shape every time: `runtime_timeout`, latency
pinned within ~0-200ms of whatever the configured wall was. There is no
partial-progress signal anywhere in these 4 trials — no `model_valid`,
`model_invalid`, or `fallback_output`, and no latency meaningfully short of
the wall that would suggest the search was "close" to finishing when killed.
A second, independent record (`held_out_dual_card_01`) at 60s shows the same
pattern, so this is not one pathological record.

If the 30s wall were merely a little short, extending it should have let at
least one trial's search finish early — showing latency well under the new,
longer wall, or a decode_outcome other than `runtime_timeout`. That never
happened across 3x the original budget. **This is evidence the compiler-tree
witness search is genuinely unbounded/stuck on this checkpoint+grammar for
these records, not that the eval protocol's timeout is merely miscalibrated.**
A defensible "bump the wall to 60s" eval-config change is not supported by
this evidence — nothing in the tested range shows convergence.

## Ship-gate check

Not applicable — n=1 per trial, isolated diagnostic slices, no `--ship-gates`
scoreboard claim.

## Decision

**Nowhere close.** Within the tested budget (30/60/90s, 2 records), the wall
extension produced zero convergence signal. Combined with the three
prior-rejected levers (decode plumbing, corpus size, step count), this closes
out the practical options for "make this checkpoint complete decode within a
modestly longer wall" — the search itself needs an algorithmic bound
investigation, which is harness-code work out of scope for this diagnostic.

## Named next lever

1. **If this thread continues:** profile *where* the compiler-tree witness
   search spends its unbounded time on a `held_out` record (not another cache
   micro-optimization — PRs #1189-#1195 already exhausted that layer) and add
   a search-depth/node-budget cap that fails closed to a diagnosable state
   instead of running until SIGALRM. This is `improve-openui-harnesses` work,
   not an eval-recipe lever, and needs its own dedicated, carefully-scoped
   change.
2. **Recommended: conclude the `great-dirac` decode/training thread here.**
   Four independent lever families (decode-plumbing micro-optimizations,
   corpus size, step count, decode-timeout-wall extension) have now all been
   tried against the same `held_out` baseline with the same result:
   `meaningful_program_rate` stays at 0.0, and this probe shows no convergence
   signal at all as budget grows. The next autotrain iteration should pick an
   **entirely different pipeline phase** — e.g. preference/surrogate-DPO
   (`references/preference.md`), an annotations-export cycle
   (`references/annotations.md`), or an experiment matrix unrelated to this
   checkpoint/grammar (`references/experiments.md`) — rather than a fifth
   attempt at this same checkpoint's decode completion.

## Validation

```text
python -m scripts.repo_policy
# repo-policy: ok (tracked + untracked)

python -m scripts.verify_version_stamps --check
# version-stamps: ok (vs HEAD; 0 changed file(s), 0 component(s) touched)

python -m scripts.verify_decode_invariants
# exit 0, agent_surfaces/canonical_defaults/strict_policies/weakening_levers unchanged
```

No harness/metric/gate/matrix source file changed this session — 4
`evaluate_model` CLI invocations against an existing checkpoint, each a
separate bounded Bash call. 0 version-stamp component bumps required.

## Scope note

- Diagnostic re-measurement only. No `--ship-gates` scoreboard claim, no
  checkpoint promotion, no `MODEL_CARD.md` update — checkpoint unchanged from
  PR #1198's session.
- `outputs/runs/diag_wall{30,60,90}_off0_s256_seed47/` and
  `outputs/runs/diag_wall60_off1_s256_seed47/` are gitignored, not committed
  — this doc is the durable record.
- `DEFAULT_DECODE_TIMEOUT_SECONDS` in `src/slm_training/levers.py` was **not**
  modified — this is measurement only, per the task's explicit instruction.

Captured: 2026-07-28T13:56:00Z
