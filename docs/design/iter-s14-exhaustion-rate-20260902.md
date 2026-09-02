# S14 / N12: LAVE stall termination — admit-probe rejection runs and the `GenerationExhausted` rate

**Honesty:** fixture-scale telemetry audit on one 4-CPU, CPU-only box shared
with other agents. **Fixture-demo, not ship.** No quality, promotion, parse,
fidelity, or readiness claim is made or implied; no gate is evaluated and none
is changed. Production decode behaviour is unchanged — the only code change is
two read-only counters.

Machine-readable sidecar:
[`iter-s14-exhaustion-rate-20260902.json`](iter-s14-exhaustion-rate-20260902.json).

## Hypothesis under test

LAVE (arXiv:2602.00612) recovers from **τ consecutive proposal failures** by
replacing the context with a cached prefix. **N12** claims this repository
already covers that recovery through `asap.penalize`
(`src/slm_training/models/twotower.py`, MaskGIT admit-probe rejection path) and
`remask_ratio`, so the only measurable gap left is *stall termination* — the
`GenerationExhausted` rate (`src/slm_training/web/service.py`) — which N12
asserts is ≈ 0 on fixtures.

**Falsifier:** a non-trivial `GenerationExhausted` rate.

## Checkpoint (named, as required)

The committed fixture checkpoint is
`src/slm_training/resources/checkpoints/playground_demo/last.pt`. It **cannot be
loaded by current code**: `TwoTowerModel.from_checkpoint` raises

```
OutputContractError: checkpoint output contract v0 is incompatible with
required symbol_only/v2; retrain from symbol-only targets
```

(`require_current_output_contract`, `src/slm_training/dsl/language_contract.py`),
and there is no migration path. Following the precedent set by
[`compiler-decode-cost-20260902.md`](compiler-decode-cost-20260902.md), the
measurement therefore uses a **scratch twin of the same architecture**:
`d_model` 96, 2 context + 3 denoiser layers, `gen_steps` 8, scratch context
backend, lexer output, contract v2 — 787,586 parameters (the committed fixture
has 740,352). Trained with AdamW `lr=3e-3` for 60 steps (36 s) on the 13
contract-eligible records of the smoke96 suite (`from_records` rejects the 11
`Semantic roles:` prompts and role-unsafe targets in that file). Session
scratch, not committed. The twin is **undertrained**, which biases the
exhaustion measurement toward *more* stalls, not fewer.

## Decode seam

Every document is decoded through `PlaygroundService.generate(prompt,
grammar_constrained=True, max_attempts=3)` with an injected `model_factory`,
wrapped in `collect_decode_stats()`. That is the seam that owns
`GenerationExhausted`, `SubstitutedGeneration`, and the retry/validate policy.

`PlaygroundService._load_locked` pins the serving policy — including
`grammar_ltr_primary = True` and `compiler_decode_mode = "tree"` — so the
shipped serving path **is not the MaskGIT path and fires no admit probes at
all** (`admit_probe_canvases = 0` on every serving-default run below). Two arms
were therefore run:

| arm | decode lane | admit probes |
| --- | --- | --- |
| `serving_default` | as `_load_locked` pins it (compiler-tree LTR) | none |
| `maskgit` | lane re-pointed **after** load: `grammar_ltr_primary=False`, `compiler_decode_mode="off"`, `grammar_fastpath_mode` ∈ {`mask`, `hybrid`}, `unmask_mode="positions"` | on |

Both arms keep every serving honesty flag `_load_locked` sets
(`allow_unconstrained_fallback=False`, `grammar_finalize_validate=False`,
`generate_max_attempts=3`). `torch.set_num_threads(2)`, device `cpu`, every run
wrapped in `timeout 170` (`MAX_RUN_MINUTES = 3`). No run timed out; there are no
timeout rows.

Suites (first *n* records in file order):
`e938_role_safe_all_targets_smoke96_v1/suites/{smoke,held_out}` and
`e938_role_safe_all_targets_v2/suites/adversarial` (that suite contains only 4
records, so n=4 is the whole suite).

## Measured results

Real `admit_fill` (no synthetic rejection). `runmax` = maximum consecutive
admit-probe rejections with no intervening commit, over the whole decode.

| arm | suite | n | seed | probes | probes w/ committed suffix | rejections | runmax | asap penalties | docs w/ certified fallback | **exhausted** | s/record |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| maskgit (`mask`) | smoke | 10 | 0 | 663 | 663 | 0 | 0 | 0 | 5 | **0** | 4.161 |
| maskgit (`mask`) | held_out | 10 | 0 | 1003 | 1003 | 0 | 0 | 0 | 5 | **0** | 5.042 |
| maskgit (`mask`) | adversarial | 4 | 0 | 466 | 466 | 0 | 0 | 0 | 4 | **0** | 6.379 |
| serving_default | smoke | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | **0** | 2.005 |
| serving_default | held_out | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | **0** | 2.493 |
| serving_default | adversarial | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | **0** | 3.408 |

Replicates (must be identical — attempt 1 is deterministic, `grammar_sample_decode`
only turns on from attempt 2, and no record needed a second attempt):

| replicate | probes | rejections | runmax | certified fallbacks | exhausted | s/record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke, maskgit, `grammar_fastpath_mode=hybrid`, seed 0 | 663 | 0 | 0 | 5 | 0 | 4.516 |
| smoke, maskgit, `mask`, **seed 1** | 663 | 0 | 0 | 5 | 0 | 5.507 |

Diagnostic control arms — `admit_fill` is **synthetically** forced to reject
(clearly labelled: these are not measurements of the model):

| control | probes | rejections | runmax | certified fallbacks | exhausted | s/record |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke, 75 % synthetic rejection | 1943 | 1458 | 3 | 6 | **0** | 6.021 |
| smoke, 100 % synthetic rejection | 3010 | 3010 | **306** | 5 | **0** | 3.428 |

Totals over the six real runs: **48 documents, 2,132 admit probes, 0
rejections, maximum consecutive-rejection run 0, 0 ASAp penalties, 0
`GenerationExhausted` (rate 0.000), 28 documents answered with a certified
fallback.**

## What the numbers say

1. **The falsifier did not fire.** The `GenerationExhausted` rate is 0/48 on
   both arms and all three suites, and stays 0/10 even when *every* admit probe
   is synthetically rejected and the loop runs a 306-long consecutive-rejection
   run. Left-to-right repair absorbs the stall before the service's attempt
   budget is exhausted.
2. **N12's rationale is not supported by measurement.** `asap.penalize` fired
   **0 times** on every real run: `asap_decode` defaults to `False`
   (`src/slm_training/harnesses/model_build/config.py:238`), so the `AsapLedger`
   is never constructed on the production path. `remask_ratio` is `0.0`
   (`config.py:209`), so remasking never ran either. Neither of the two
   mechanisms N12 credits with "already covering LAVE's recovery" executed a
   single time.
3. **There was nothing to recover from.** The admit probe rejected 0 of 2,132
   real probes. Every one of those 2,132 probe canvases carried a committed
   token *after* the first hole
   (`admit_probe_committed_suffix == admit_probe_canvases`) — exactly the HX1
   left-prefix span `admit_fill` cannot validate — so on this configuration the
   probe is a pure over-approximation and structurally cannot reject.
   `pick_constrained_token` has already filtered the proposal against the left
   prefix, and the probe adds no further constraint.
4. **The live failure channel is certified substitution, not exhaustion.**
   28 of 48 documents (14 per arm) ended with `certified_fallbacks >= 1` — a
   certified minimal program substituted for a decode that did not produce a
   valid program — and `PlaygroundService` recorded **every one of them as a
   successful generation**. `_raise_on_substituted_generation` detects this via
   `consume_generation_evidence`, which only the ONNX backend exposes; the torch
   backend does not, so `SubstitutedGeneration` was raised 0 times. Under I6
   that substitution is a failed decode, and the exhaustion metric cannot see
   it.

## Verdict

**N12's conclusion HOLDS; N12's rationale does NOT.**

- Holds: the stated falsifier (a non-trivial `GenerationExhausted` rate) did not
  fire — 0/48 on fixtures, and 0/10 under a 100 % synthetic rejection control.
  A LAVE τ-restart lever has no observable trigger on this evidence, and this
  approach is closed for now.
- Does not hold: the claim that `asap.penalize` + `remask_ratio` "already cover"
  LAVE's recovery is unsupported. Both are config-gated **off** by default and
  executed zero times; the coverage is dormant, not demonstrated. The
  approach is closed by *absence of a trigger*, not by *presence of recovery* —
  the goal of not stalling remains open and is now measurable.

## Proposed lever (preregistered, NOT implemented)

`N12-tau` — **`tau_admit_reject_restart`**: after τ consecutive admit-probe
rejections with no commit, restart the MaskGIT sweep from a cached certified
prefix instead of continuing to propose into a dead canvas.

- **Trigger metric:** `DecodeStats.admit_probe_reject_run_max` (added by this
  card, read-only).
- **Preregistered gate before any implementation:** do not implement until a
  corpus is found where `max_consecutive_rejection_run >= 2` on at least 10 % of
  documents with real `admit_fill`, n ≥ 96, two seeds. On the present evidence
  that metric is 0 on 100 % of documents, so the lever would be a parameter
  tuned against an unmeasured effect.
- **Arms if the gate is met:** control (τ off) / τ = 4 / τ = 16.
- **Endpoint:** `GenerationExhausted` rate **and**
  `documents_with_certified_fallback` (the second is the channel that actually
  moves).
- **Family:** decode-recovery. **Invariants:** I1–I6 unchanged; a restart must
  re-verify before commit (I3) and may never bypass the deterministic/singleton
  path (I2).

Nothing above is implemented in this change.

## Higher-priority gap found on the way (reported, not fixed)

`N12-sub`: on the torch backend, a certified substitution is invisible to the
serving harness. 28/48 documents were substituted and all 28 were persisted as
successful generations. `_raise_on_substituted_generation`
(`src/slm_training/web/service.py`) is a no-op for any backend without
`consume_generation_evidence`. The `DecodeStats.certified_fallbacks` counter
already carries the signal; nothing reads it at the service seam. This is a
larger honesty gap than the stall-termination question N12 asked about, and it
is the natural successor card.

## Telemetry added (read-only, off by default)

`DecodeStats.admit_probe_rejections` and
`DecodeStats.admit_probe_reject_run_max`
(`src/slm_training/models/decode_stats.py`), incremented by
`TwoTowerModel._note_admit_rejection` in the positionwise MaskGIT unmask lane
(`src/slm_training/models/twotower.py`). No decode path reads either field back,
and with no `collect_decode_stats()` collector active `get_active_stats()` is
`None` and nothing is written. `admit_probe_reject_run_max` merges as a maximum,
never a sum.

`tests/test_models/test_admit_rejection_telemetry.py` proves the decode is
byte-identical with and without a collector attached, that the counters are 0
when every probe admits, that the whole rejection sequence is one run when no
probe admits, and the merge semantics.

**Not instrumented (stated so the numbers are not over-read):** the cluster
unmask lane (`unmask_mode="cluster"`), the one-hole exact-bypass admit probe,
and the grammar stream hard-error remask path. All three are off or unreached in
the configurations measured here.

## Reproduction

```
PYTHONPATH=$PWD/src .venv/bin/python   # torch.set_num_threads(2), device cpu
timeout 170 <runner> --suite {smoke,held_out,adversarial} \
    --arm {maskgit,serving_default} --n {10,10,4} --seed {0,1} \
    --fastpath-mode {mask,hybrid} [--stress-reject-mod M]
```

Scratch runners (session scratch directory, not committed): `train_twin.py`
(scratch twin), `n12_run.py` (per-suite decode + telemetry), `mkjson.py`
(aggregation into the sidecar JSON).
