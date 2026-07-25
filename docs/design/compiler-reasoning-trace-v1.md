# CompilerReasoningTraceV1 — target-trace contract (LOT0-02 / SLM-249)

**Machine-readable contract:** [`compiler-reasoning-trace-v1.json`](compiler-reasoning-trace-v1.json).
**Bounded K/c coverage evidence:** [`compiler-reasoning-trace-coverage/coverage_report.md`](compiler-reasoning-trace-coverage/coverage_report.md)
(evidence class `bounded_probe`, n=16 — see that report before treating any number
here as corpus-derived).

This is the LOT0-02 deliverable named by the LOT0-01 authorization's
(`docs/design/lotus-openui-fidelity-contract-v1.md`) `allowed_lot1_work`:

1. Define the OpenUI target-trace contract.
2. Specify the K/c/R sweep grid, the unlooped extra-update control, and the
   replace/no-injection ablation as a preregistered **fixture-scale plan (no
   training)**.
3. Extend `causal_trace.py`'s counterfactual-replay machinery only insofar as
   needed to **define** (not run) a `causal_latent_use` falsification test.
4. Bounded target-contract probes (reading/inspecting a handful of existing
   compiler/decision traces; no corpus build).

Nothing below authorizes model code, training, or a corpus build. Verdict:
**`inconclusive`** — see [Gate verdict](#gate-verdict).

## Stage set (data-derived, not LOTUS-by-analogy)

The six stages are not inherited from LOTUS's `K=6` by analogy. They map 1:1 onto
concepts `SemanticPlanV1` (`src/slm_training/data/progspec/semantic_plan.py`)
already extracts deterministically from an OpenUI AST, plus one stage sourced from
the production codec's canonical statement order:

| step_kind | concept | reused source |
| --- | --- | --- |
| `intent_contract` | target kind/category, root intent/archetype, required output contract | `SemanticPlanV1.archetype` |
| `component_inventory` | unordered required component families and optional/ambiguous status | `SemanticPlanV1.role_slots` |
| `topology_skeleton` | canonical/accepted tree skeleton without surface lexemes | `SemanticPlanV1.topology.parent_relation_candidates` (anonymized) |
| `semantic_edges_roles` | component ownership, list occupancy, required semantic roles | `SemanticPlanV1.topology.parent_relation_candidates` + `sibling_order_groups` |
| `binding_scope_references` | binder owners, arities, pointer/reference identities, scope obligations | `SemanticPlanV1.symbols` / `SemanticPlanV1.bindings` |
| `serialization_plan` | canonical ordering/realization commitments needed to produce a legal program | `production_codec.statement_binding_order` |

No stage is invented beyond what the repository's existing extractors already
compute; this issue creates no new compiler, parser, or validity definition.

## Typed step schema

`TraceStepV1` (`src/slm_training/data/progspec/compiler_reasoning_trace.py`) is a
frozen, `extra="forbid"` pydantic model carrying every field the issue requires:
`trace_version`, `record_id`/`source`/`split`/`fingerprint`, `step_index`,
`step_kind`, `canonical_target`, `accepted_targets`, `set_valued_fields`,
`partial_order`, `support`/`confidence`/`ambiguity_reason`, `is_truncated`,
`source_refs`, `surface_tokens`, and `target_ids` (always `None` here — reserving
trace tokens in a shared vocabulary is model-implementation work gated behind a
reviewed LOT1 contract, not this issue).

`CompilerReasoningTraceV1` wraps an ordered tuple of steps plus a `stage_order`;
`validate_no_cycles` fails closed on any cyclic `partial_order` dependency, and a
partial trace (curriculum stage < K) is simply a strict prefix of `stage_order` —
`is_partial` makes that explicit rather than silent.

## Visible trace representation

`serialize_visible_trace`/`parse_visible_trace` produce a lossless, round-tripping
`<trace_v1><step:KIND>{...canonical compact JSON...}</step>...</trace_v1>`
representation — typed delimiters plus compact tokens, separate from the final
OpenUI program surface, not optimized for human prose.

## Extraction

`extract_compiler_reasoning_trace` (`src/slm_training/data/semantic_plan/
compiler_reasoning_trace_extract.py`) reuses `OpenUISemanticPlanExtractor` and
`production_codec.parse_statement_bindings`/`statement_binding_order` — no
evaluator or model output enters any target field; extraction is a pure function
of an already-validated `ProgramSpec`/`DslPack`, and is deterministic (verified by
repeated-call equality tests).

## K/c budget analysis (bounded probe)

`scripts/report_compiler_reasoning_trace_coverage.py` extracts a trace for every
record in the repo's existing 16-record hand-authored fixture set
(`src/slm_training/resources/test_seeds.jsonl`) and reports per-stage
`canonical_target` length percentiles. Full report:
[`compiler-reasoning-trace-coverage/coverage_report.md`](compiler-reasoning-trace-coverage/coverage_report.md).

- proposed K: **6** (fixed by stage-set construction, not data-derived truncation)
- proposed c (provisional, chars): **479** (max per-stage p95 over n=16)
- fraction of probe records requiring all 6 stages: **1.000**
- fraction truncated: **0.000**

This is explicitly **not** a corpus-scale K/c statistic — the LOT0-01 authorization
blocks corpus generation for this issue. A future issue that runs the oracle
ceiling experiment below against a real corpus is the place to re-derive K/c from
production-scale data.

## Oracle ceiling experiment — plan only, not run

Full arm/matching/measurement specification lives in
`compiler-reasoning-trace-v1.json#oracle_ceiling_experiment_plan`. Summary:

| arm | description |
| --- | --- |
| `causal_baseline` | prompt → final program, no trace exposure |
| `explicit_gold_trace` | prompt + visible gold trace → final program |
| `shuffled_wrong_trace` | prompt + `make_shuffled_control(trace)` → final program |
| `partial_trace_by_stage` | prompt + `make_partial_trace(trace, s)` for `s` in 1..K |
| `accepted_alternative_trace` | prompt + a non-canonical accepted-target trace (generator not yet built — arm specified, not implemented) |
| `equivalent_length_nuisance_control` | prompt + a token-length-matched non-semantic filler span |
| `existing_plan_baseline` | prompt + raw `SemanticPlanV1` JSON, to rule out "any structured plan helps" |

Matching rules: same parent checkpoint/from-scratch policy, same program targets
and optimizer/schedule/data order/checkpoint-selection rule across arms; extra
trace-token exposure reported separately; a continued baseline receives matched
additional tokens/updates; minimum three paired seeds before any ceiling claim.

**Status: `planned_not_run`.** Running any arm requires GPU training and a corpus
beyond the bounded 16-record probe — both explicitly blocked by the LOT0-01
authorization for this issue.

## `causal_latent_use` falsification test — defined, not run

`CausalLatentUseFalsificationSpecV1` / `describe_causal_latent_use_falsification_test`
(`src/slm_training/models/causal_trace.py`) define the test: force the future K×c
latent workspace to a null/shuffled value at a fixed loop iteration (the same
forced-counterfactual shape as `CausalLMOpenUIPlugin.replay_causal_action`), hold
every other input identical, and check whether the downstream strict-semantic
outcome changes. If it does not (paired, matched seeds), causal use is falsified —
LM-head decodability of a latent state is not itself evidence of causal use. This
spec has no model dependency and executes nothing; it cannot run until LOT1
(`SLM-250`) exists.

## Gate verdict

`CompilerReasoningTraceGateV1` (`compiler-reasoning-trace-v1.json#gate`):

```text
verdict: inconclusive
k_c_recommendation: {k: 6, c_chars: 479, evidence_class: bounded_probe, sample_size: 16}
accepted_stage_set_order: [intent_contract, component_inventory, topology_skeleton,
                            semantic_edges_roles, binding_scope_references,
                            serialization_plan]
allowed_lot1_implementation: none — SLM-250/LOT1-01 stays gated on a positive
                              oracle-ceiling result, which requires running the
                              plan above (not authorized by this issue)
blocked_claims: no semantic-quality claim; no corpus-derived K/c claim;
                 no causal_latent_use claim; no GPU-training claim
```

The contract, extractor, and serialization are real and tested. The one thing that
would let `SLM-250` (LOT1's actual K×c model implementation) proceed — a run of the
oracle ceiling experiment above — is out of scope here by design. Declaring
`oracle_ceiling_positive`, `no_downstream_ceiling`, or
`explicit_trace_equivalent_to_existing_plan` without running that experiment would
be exactly the "no claim that explicit trace benefit implies latent benefit"
failure the issue's acceptance criteria forbid. `inconclusive` is the only honest
verdict pending that campaign.
