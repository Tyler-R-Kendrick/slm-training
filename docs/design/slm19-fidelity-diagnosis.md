# SLM-19 diagnosis — matched integrated-corpus fidelity

Status: **diagnosis complete; SLM-19's original acceptance is already met by the
final P13 control; the real blocker is re-scoped below.**

Evidence: [data-synthesis.md](data-synthesis.md) ·
[data-synthesis-results.json](data-synthesis-results.json) ·
[iter-e261-gold-ast-counterfactual-corpus-20260716.md](iter-e261-gold-ast-counterfactual-corpus-20260716.md) ·
[iter-remediated-roots-128step-diffusion-fidelity2-20260715.md](iter-remediated-roots-128step-diffusion-fidelity2-20260715.md).

## 1. The original premise is stale — the blocker moved

SLM-19 was filed on the early **E0** probes, which showed
`placeholder_fidelity = 0.0` on `held_out` + `rico_held` for both the fixture and
integrated corpora. The **final P13 matched control** (identical E50 recipe — CPU
scratch, 80 steps, batch 4, lr 3e-4, seed 0, honest slot contract, no
template-fill, best-of-1, same 5 held_out / 5 rico; only the corpus changes)
already resolved it:

| Arm | held_out fidelity | rico_held fidelity |
| --- | ---: | ---: |
| Fixture control (25 rows) | 0.08 | 0.067 |
| **Integrated (176 rows)** | **0.12** | **0.10** |
| Δ | **+0.04** | **+0.033** |

The integrated corpus **strictly beats** fixture on **both** suites and is **> 0 on
both** — exactly SLM-19's acceptance criterion. That is why it was reasonably
canceled: as written, it is satisfied.

## 2. The real blocker migrated to parse / meaningful-program rate = 0.0

Both checkpoints still fail ship gates — but on `parse_rate` /
`meaningful_program_rate = 0.0`, **not** fidelity. The eval ceiling
(gold-as-prediction) is **1.0** on every suite, so the targets are correct and
reachable; the model cannot reach them at this scale.

## 3. Root cause — a CPU-smoke *scale* ceiling on two axes (not a data defect)

1. **The corpus is tiny by configuration, not capability.** The "176-row
   integrated corpus" was built with `--programspec-count 1 --rico-limit 1` — 176
   deterministic derivatives of **one** generated program + **one** RICO screen +
   fixtures, and it is mostly **Bronze (153/176)**. Too small and structurally
   narrow to teach generalizable placeholder binding — hence "only small
   structural changes." `ProgramGenerator.generate(count)` +
   `generate_until_covered` + `CoverageTracker` can produce arbitrarily many
   diverse programs.
2. **Compute is far below threshold.** 80 CPU scratch steps / batch 4 from a
   from-scratch denoiser cannot form meaningful programs on unseen layouts;
   structure moves (structural_similarity 0.16–0.57) but `meaningful_program_rate`
   stays 0.
3. **Decode / loss levers are exhausted.** E261: *"syntax success from the
   deterministic constrained layer cannot substitute for semantic fidelity."*
   remediated-roots: *"stronger fidelity loss [0.5→2.0] did not improve either loss
   or fidelity → reject the checkpoint."*

## 4. Re-scoped path to a shippable signal (smallest → largest lever)

1. **Scale the generated corpus** — `--programspec-count` in the hundreds–thousands
   + full RICO; verify diversity via the coverage report + family/structure
   histograms.
2. **Re-run the matched control at real compute** — HF frozen context (SmolLM2) +
   more steps + GPU (`scripts/hf_jobs_train.py` / `remote_train.py`), not 80 CPU
   scratch steps. Ceiling = 1.0 ⇒ capacity is the limiter.
3. **Keep the honest slot-contract / inventory-in-prompt recipe** (E35/E53) for the
   fidelity arm — every champion that ever cleared used it; plain generation cannot
   emit exact held-out namespaces.
4. **Add a preference / RL stage** on the committed **E261 semantic-preference
   corpus** (239 events) after SFT — the documented lever for semantic fidelity
   beyond decode.
5. **Stop tuning decode constraints / fidelity-loss weight** — the evidence shows
   they do not move semantic fidelity.

## Decision

SLM-19 as written is met by the final P13 E50 control. Its remaining spirit —
getting fidelity to a *shippable* level — is a **corpus-scale + real-compute +
preference-stage** effort, not a data-pipeline fix, and belongs on the
[promotion-pipeline](promotion-pipeline.md) track. No ship claim is made here.
