# Local decision interventions for TwoTower

**Status:** research intake plus the measured E248–E289 local-decision campaign,
recommitted as the LDI0 evidence contract (LDI0-01). E249 is rejected
(constraint-shadow ranking generalized locally but regressed semantic quality on
every suite), and the chain through E284 establishes the **current blocker**:
stable grammar-state support is necessary but not sufficient for the FTPO
objective, because independently judged legal alternatives produce different
bad-token sets at the same state. This document is a contract and research
synthesis; it ships no event-schema or trainer code, no checkpoint or adapter, and
**no model-quality or ship claim**.

## Source and audit

The source was the public ChatGPT share
[`6a593158-85c4-83ea-80b1-b6fb893b26bc`](https://chatgpt.com/share/6a593158-85c4-83ea-80b1-b6fb893b26bc).
The normal page reader returned only the application shell, so the server-rendered
conversation payload was decoded and its citation records were normalized against
primary arXiv and OpenReview pages. The reviewed inventory is committed as
[`local-decision-sources.json`](../../src/slm_training/resources/autoresearch/local-decision-sources.json):
34 distinct academic works and eight implementation/documentation sources. The
DeepSeek-R1 Nature DOI and the two OpenReview URLs are retained as alternate URLs
rather than double-counted papers. Twenty-five works came from the original share;
LDI0-01 added nine required prior-art works, each labeled **Adjacent** —
inventoried, not reimplemented: LoRA and AdaLoRA extend *Adapter actuators*; PCGrad
and MGDA add *Multi-objective balancing* (trade the correction loss against the
locality tether); PICARD, Grammar-Aligned Decoding, and Min-p add *Constrained
decoding and sampling* (legality only, not preference); DeepSeekMath/GRPO extends
*Verifiable training*; and TAB-PO extends *Local preference*.

| Cluster | Sources | Relevance here |
| --- | --- | --- |
| Local preference | [Unlikelihood](https://arxiv.org/abs/1908.04319), [DPO](https://arxiv.org/abs/2305.18290), [TDPO](https://arxiv.org/abs/2404.11999), [TIS-DPO](https://arxiv.org/abs/2410.04350), [ConfPO](https://arxiv.org/abs/2506.08712), [TGDPO](https://arxiv.org/abs/2506.14574), [Antislop](https://arxiv.org/abs/2510.15061), [TokenRatio](https://arxiv.org/abs/2605.12288) | Correct the exact action at a fixed state; do not spread preference loss across an entire program. |
| Credit assignment | [SCAR](https://arxiv.org/abs/2505.20417), [Discriminative Policy Optimization](https://arxiv.org/abs/2505.23363) | A delayed failure is not evidence that an earlier token caused it. Require immediate verifier evidence or a counterfactual continuation. |
| Verifiable training | [Structured Output with Schema RL](https://arxiv.org/abs/2502.18878), [DeepSeek-R1](https://arxiv.org/abs/2501.12948), [Minerva](https://arxiv.org/abs/2602.00513) | Grammar and contract checks can label events, but local supervised correction does not bypass RL readiness. |
| Adapter actuators | [DoRA](https://arxiv.org/abs/2402.09353), [PiSSA](https://arxiv.org/abs/2404.02948), [MoLoRA](https://arxiv.org/abs/2603.15965), [PermDoRA](https://arxiv.org/abs/2606.11262) | Adapter form, initialization, and routing are separate from the event/objective. They remain causal-track follow-ups. |
| Representation interventions | [ReFT](https://arxiv.org/abs/2404.03592), [interpretable SAEs](https://arxiv.org/abs/2309.08600), [SAE scaling](https://arxiv.org/abs/2406.04093), [Gemma Scope](https://arxiv.org/abs/2408.05147) | SAEs may diagnose states later; they are not semantic oracles and are unnecessary for the first direct-logit test. |
| SAE steering dispute | [AxBench](https://arxiv.org/abs/2501.17148), [SAE rebuttal](https://arxiv.org/abs/2605.31183) | Conflicting results make a matched direct-supervision control mandatory before any SAE steering claim. |
| Locality preservation | [AlphaEdit](https://arxiv.org/abs/2410.02355), [AlphaEdit reproducibility](https://arxiv.org/abs/2606.26783) | Measure untouched-logit and end-to-end drift directly; an apparently local target does not prove a local update. |

The implementation references are Auto-Antislop, Antidoom, Hugging Face PEFT
LoRA/trainable-token documentation, TransformerLens, SAELens, NNsight, and pyreft.
Auto-Antislop was reviewed at `8fb98fdf019e6fcc20164f9bdec41f9008fcd632`;
no license file was present, so its source is not copied. Antidoom was reviewed at
`bd6a126476e18554b0cacaea3fd9f258fdde1f97` under Apache-2.0.

## Existing seam and missing evidence

E228 already constructs a compiler decision canvas, restricts logits to legal
candidates, and applies cross-entropy plus a strongest-alternative margin. E228 and
E229 did not clear the unchanged ship gates. The useful gap is therefore narrower
than another trainer family:

1. E228 samples gold compiler states, not actual policy failure onsets.
2. It assumes one gold action and does not represent separately verified good and
   bad action sets at the same state.
3. It has no frozen-reference locality tether or explicit drift telemetry.
4. It has no event-family balancing or immutable event/checkpoint identity.

The existing preference harness remains the evidence and training owner. The
append-only decode trace remains the observation owner. The quality matrix and
autoresearch remain the bounded-experiment owner. No second orchestration or
training stack is introduced.

## Separation of concerns

Evidence, objective, actuator, experiment, and promotion are distinct and
separately owned. No single change may collapse them.

- **Evidence** — exact-state decision events live in the preference harness
  (`DecisionEventV1` in [`harnesses/preference/local_decisions.py`](../../src/slm_training/harnesses/preference/local_decisions.py)).
  Constraint shadows certify decoder legality only; they never become semantic
  preference labels without same-state counterfactual verification.
- **Objective** — local losses ([`harnesses/preference/local_train.py`](../../src/slm_training/harnesses/preference/local_train.py))
  consume events; they do not define their own event or trace format.
- **Observation** — the append-only decode trace
  ([`harnesses/distill/trace_store.py`](../../src/slm_training/harnesses/distill/trace_store.py))
  owns replayable trajectories with immutable checkpoint/decode identity.
- **Actuator** — LoRA/DoRA/PiSSA, removable TwoTower deltas, ReFT, and SAE are
  actuator *choices*, not event schemas. SAE work stays behind matched direct
  baselines.
- **Experiment / promotion** — the quality matrix and autoresearch
  ([`scripts/run_quality_matrix.py`](../../scripts/run_quality_matrix.py)) own
  bounded experiments and gates. Hard grammar/compiler constraints remain deployed;
  interventions aim to reduce error mass, not replace deterministic guarantees. New
  experiments use the `LDI` campaign name in prose/config but draw globally unique
  E-IDs from the existing allocation process — no E-ID is reserved or assumed here.

The first new contract is `DecisionEventV2` / action-table evidence; the first
actuator experiments are causal PEFT and removable TwoTower deltas.

## Decision event contract

`DecisionEventV1` identifies one exact masked-token decision. It stores the
formatted context, pre-decision canvas, position, legal/good/bad token IDs, evidence
kind and confidence, source/group/split, and checkpoint/tokenizer/decode hashes.

Two evidence paths are allowed:

- **Constraint shadow:** the raw argmax is outside the verifier's allowed set and
  the constrained selection is inside it. This is exact grammar evidence, not a
  semantic-quality label.
- **Counterfactual verification:** replay candidate actions from the identical
  state, continue under the same policy/seed, and label only candidates whose
  resulting program passes the named verifier. This is required for semantic or
  multi-good/multi-bad events.

Final-output failure alone never creates a token label. Splits group by prompt and
record family so related counterfactuals cannot cross train/held-out boundaries.

## Objectives and locality

For good action `g`, bad action `b`, and decision logits `z`, define
`delta(g,b) = z[g] - z[b]`. The clipped FTPO adaptation is:

```text
w(g,b) = clamp((epsilon - delta(g,b)) / epsilon, 0, 1)
L_ftpo = mean(w(g,b) * softplus((epsilon - delta(g,b)) / tau))
```

The set form averages only verifier-backed `G × B` pairs. A separate frozen parent
produces reference logits on the same state. Non-target vocabulary logits receive a
strong MSE tether; target logits receive a weaker MSE only outside a grace band.
This is an independently implemented TwoTower adaptation of the published behavior,
not a faithful reproduction of Auto-Antislop, DPO, TGDPO, or TokenRatio.

The first actuator updates TwoTower's existing trainable parameters. It is a local
loss with a global parameter update, not a removable LoRA. LoRA/DoRA/PiSSA, ReFT,
SAE discovery, adapter routing, iterative remine, and RLVR remain explicit follow-up
hypotheses.

## Proposed V10 campaign

All rows use the same parent, immutable event file, split, step count, learning
rate, and seed. Set-valued rows fail closed if the corpus contains no verified
set-valued event.

| ID | Intervention | Purpose |
| --- | --- | --- |
| E248 | Parent, no update | Matched eval-only control. |
| E249 | Event CE + margin | E228-style objective on exact events. |
| E250 | Bad-token unlikelihood | Simplest localized negative control. |
| E251 | Single-good/single-bad FTPO | Tests clipped exact-state preference. |
| E252 | Verifier-backed set FTPO | Tests multiple verified alternatives. |
| E253 | E252 + frozen-reference tether | Tests locality preservation. |
| E254 | E253 + balanced sampling | Tests source/kind/rejected-token imbalance. |

Initial borrowed hypotheses are `epsilon=2`, `tau=1`, non-target tether `0.4`,
target tether `0.05`, and target grace `1.0`. They are unvalidated for TwoTower.

Event metrics are good/bad probability mass, chosen-win, margin-win, mean/median
margin, active-weight fraction, and held-out recurrence by decision kind. Locality
metrics are non-target MSE, target excess MSE, unchanged-decision rate, and
full-vocabulary drift. End-to-end authority remains the unchanged five-suite
scoreboard and ship gates.

Falsify a row if it does not improve held-out event metrics, exceeds the matched
reference-drift budget, or regresses any protected ship gate. E248 (control) and
E249 (CE + margin) are measured; E250–E254 remain unrun and fail-closed. There is
no intervention checkpoint, model-card update, or promotion. Further LDI rows draw
new globally unique E-IDs from allocation rather than reserving numbers here.

## Measured event-mining prerequisite

The production strict compiler-tree decoder now emits exact branching states rather
than reconstructing them from final strings. The committed
`e249_constraint_shadows_v1` corpus contains 2,035 identity-homogeneous events from
65 document groups, split by group into 1,716 train and 319 held-out events. Its
manifest, source trace, diagnostics, and honesty boundary are recorded in
[`iter-e249-exact-event-mining-20260716.md`](iter-e249-exact-event-mining-20260716.md).

All events are grammar constraint shadows. E249 proved that they support a
matched lexical-decision experiment but not a semantic-quality objective:
held-out chosen win reached 0.7649 while structure and reward regressed on every
suite. Do not run E250/E251 on this corpus as quality labels. E252-E254 remain
fail-closed because the corpus contains no counterfactual or set-valued evidence.
Measured result:
[`iter-e249-local-ce-margin-20260716.md`](iter-e249-local-ce-margin-20260716.md).

## Measured falsification chain and current blocker

E248–E289 is the authoritative measured record; per-experiment detail lives in the
`iter-e248-*` … `iter-e289-*` notes and the V10 narrative of
[`quality-experiment-matrix.md`](quality-experiment-matrix.md). Distilled: E249
showed constraint-shadow events support a matched lexical-decision objective but
regress semantic quality on every suite, and the chain through E283/E284 established
the **current blocker** —

> Stable grammar-state support is necessary but not sufficient for the FTPO
> objective, because independently judged legal alternatives can produce different
> bad-token sets at the same grammar state.

(See [`iter-e284-signature-support-profile-20260717.md`](iter-e284-signature-support-profile-20260717.md).)
Admitting a stable support signature therefore does not by itself imply objective-
or action-partition support. The next contract (`DecisionEventV2` / action tables)
plus same-state counterfactual verification target exactly this gap. No training has
run on the corpus under a semantic objective.

## Measured status

2026-07-17 — LDI0-01 recommits this document as the local-decision evidence
contract, extends the source inventory to 34 works (nine **Adjacent** additions),
names the evidence/objective/observation/actuator/experiment owners, and records
the E249–E284 falsification chain and current blocker. Documentation and
source-inventory change only: no event-schema or trainer code, no checkpoint or
adapter, no model-card quality update, and **no model or ship claim**. `python -m
scripts.repo_policy` passes and the focused source/lineage tests are green.
