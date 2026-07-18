# Local decision interventions for TwoTower

**Status:** LDI0-01 architecture/research contract plus the measured E248-E286
local-decision campaign. The chain is negative: E249 is rejected (constraint-shadow
ranking generalized locally but regressed semantic quality on every suite), and no
local-preference intervention has cleared the unchanged ship gates. This document is
the canonical repository owner of the local-decision-intervention synthesis and
architectural boundaries; it contains no model-quality or ship claim.

## Source and audit

The source was the public ChatGPT share
[`6a593158-85c4-83ea-80b1-b6fb893b26bc`](https://chatgpt.com/share/6a593158-85c4-83ea-80b1-b6fb893b26bc).
The normal page reader returned only the application shell, so the server-rendered
conversation payload was decoded and its citation records were normalized against
primary arXiv and OpenReview pages. The reviewed inventory is committed as
[`local-decision-sources.json`](../../src/slm_training/resources/autoresearch/local-decision-sources.json):
**34 distinct academic works and eight implementation/documentation sources** — the
original 25 works plus nine added for this LDI0-01 contract (multi-objective /
gradient-conflict optimization, PEFT actuators, constrained decoding, and
token-critical preference lineage), each verified against arXiv on 2026-07-17. The
DeepSeek-R1 Nature DOI and the two OpenReview URLs are retained as alternate URLs
rather than double-counted papers, and no paper is duplicated under an alternate URL.

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

## Architecture contract (LDI0-01)

This section is the canonical architectural boundary for the local-decision program.
It exists so future agents do not recreate a parallel harness, repeat the falsified
E249-E284 chain, or treat a local-metric gain as promotion evidence. The Linear
project document is planning context; this file plus the sources manifest are the
source of truth.

**Separation of concerns (do not collapse).** Evidence, objective, actuator,
experiment, and promotion are distinct concerns with distinct owners — extend the
owner, never fork a second stack:

| Concern | Owner (do not fork) | Rule |
| --- | --- | --- |
| Exact-state decision evidence | [`harnesses/preference/local_decisions.py`](../../src/slm_training/harnesses/preference/local_decisions.py) | Events carry exact masked-token state identity. A constraint shadow certifies decoder legality only. |
| Observation / replay | [`harnesses/distill/trace_store.py`](../../src/slm_training/harnesses/distill/trace_store.py) | The append-only decode trace is the observation/replay owner; it is not a trainer. |
| Objective | [`harnesses/preference/local_train.py`](../../src/slm_training/harnesses/preference/local_train.py) | Clipped-margin FTPO and preference losses. The objective is separate from the event schema. |
| Actuator | existing trainable parameters today; later LoRA / DoRA / PiSSA / TwoTower delta / ReFT / SAE | Adapter and representation form are actuator choices, never event schemas. |
| Experiment | [`scripts/run_quality_matrix.py`](../../scripts/run_quality_matrix.py) + [`quality-experiment-matrix.md`](quality-experiment-matrix.md) | The quality matrix / autoresearch is the bounded experiment owner. |
| Promotion | ship gates + [`docs/MODEL_CARD.md`](MODEL_CARD.md) | Only the unchanged five-suite scoreboard and ship gates promote. A local metric never does. |

**Invariants:**

1. Constraint shadows certify decoder legality only. They never become semantic
   preference labels without same-state counterfactual verification.
2. Hard grammar/compiler constraints remain deployed. Interventions reduce error
   mass and improve semantics *on top of* the deterministic guarantee; they do not
   replace it.
3. The first new evidence contract is `DecisionEventV2` — an action-table extension
   of `DecisionEventV1` with stable state identity and per-action verdicts (LDI0-02).
   The first actuator experiments are causal PEFT and a removable TwoTower delta; SAE
   work stays behind matched direct-supervision baselines.
4. New experiments use the `LDI` campaign name in prose and config but obtain
   globally unique E IDs from the existing allocation process. Do not reserve or
   assume the next E number (see the E-ID rule in
   [`quality-experiment-matrix.md`](quality-experiment-matrix.md)).
5. Do not create a second orchestration or training stack. Extend the owners above.
6. No paper result is represented as a repository result. Every source is labeled
   Faithful / Adapted / Surrogate / Adjacent in
   [`research-lineage.md`](research-lineage.md) and carries an
   `implementation_status` in the sources manifest.

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

The existing preference harness remains the training owner. The append-only decode
trace remains the observation owner. The quality matrix remains the stable
experiment owner.

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

**Next contract (LDI0-02):** `DecisionEventV2` adds stable state identity and a
per-action verdict table so multiple good/bad actions at one state each carry an
explicit counterfactual verdict, targeting the objective/action-partition blocker
below. It extends `DecisionEventV1`; it does not replace the trace store or the
objective, and it is specified — not implemented — by this contract.

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
reference-drift budget, or regresses any protected ship gate. No V10/LDI
intervention row has cleared the unchanged ship gates, and there is no intervention
checkpoint, model-card update, or promotion.

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

## Measured chain and current blocker (E248-E286)

E248-E286 is authoritative and lives in
[`quality-experiment-matrix.md`](quality-experiment-matrix.md) (the V10 rows
E248-E254 and the local-preference ledger through E286) and the per-run
`iter-e2*.md` iteration docs. The chain is negative:

- **E248** — matched parent control, eval-only.
- **E249** — exact-event CE plus margin. The lexical decision objective generalized
  (held-out chosen-win 0.7649) but semantic structure and reward regressed on every
  suite; **rejected**.
- **E250-E284** — the registered bad-token, single-pair FTPO, verifier-backed set
  FTPO, frozen-reference tether, and balanced-sampling levers (and the E265-E286
  local-preference ledger) were measured; none cleared the unchanged five-suite ship
  gates or was promoted.

**Current blocker: stable state support does not imply objective/action-partition
support.** Exact-state event identity — a stable, replayable decision state — is
necessary but not sufficient. It does not establish that the good/bad *action
partition* at that state is itself verifier-supported, so a locally-improved event
metric has repeatedly failed to transfer to semantic quality under the unchanged
gates. `DecisionEventV2` counterfactual action-verdict tables (LDI0-02) target
exactly this gap. No LDI intervention row is promoted; there is no intervention
checkpoint and no model-card quality update.
