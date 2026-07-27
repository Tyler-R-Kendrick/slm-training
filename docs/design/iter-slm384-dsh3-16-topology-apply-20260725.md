# Iter: SLM-384 DSH3-16 verified operators as topology-diffusion edit actions

Date: 2026-07-25
Status: measured; execution-sparsity-supported (fixture/wiring)
Scope: bounded CPU fixture arm comparison; no checkpoint, ship, or systems claim

## Decision

DSH3-16 tests whether applying compiler-verified operator edits directly to
typed topology nodes reduces diffusion-native decode work at matched quality.
Three matched arms ran over one deterministic request set
(max 8 steps/request, enumeration budget
128 combinations/operator):

- `RECOMPUTE_ONLY` -- ordinary lowering; full typed-tree re-materialization per edit.
- `HIERARCHICAL_HEAD` -- the SLM-383 head in default `off` mode shadow-scores
  every non-singleton decision (2 scorer forwards) and always DEFERs;
  ordinary lowering applies. This measures head overhead at an unchanged workload.
- `TOPOLOGY_APPLY` -- the identical action sequence, but exact legal
  `ActionEffectV1` transitions map to topology expand/contract/move/rewrite
  edits applied directly to typed nodes (`dsl/operators/topology_apply.py`);
  non-exact or unmapped transitions defer to ordinary lowering.

Forced singletons commit with zero model forwards in every arm; compiler
expansions and pack-authority verifier calls are never skipped.

## Arm totals (matched workload)

| Arm | Steps | Node passes | Remask | Rewrite | Model forwards | Compiler expansions | Verifier calls | Direct | Deferred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `RECOMPUTE_ONLY` | 20 | 46 | 0 | 0 | 0 | 7125 | 31 | 0 | 0 |
| `HIERARCHICAL_HEAD` | 20 | 46 | 0 | 0 | 40 | 7290 | 31 | 0 | 0 |
| `TOPOLOGY_APPLY` | 20 | 40 | 12 | 8 | 0 | 7125 | 31 | 20 | 0 |

## Wall-clock / memory (not a speed claim)

| Arm | Elapsed (s) | Peak memory (bytes) |
| --- | ---: | ---: |
| `RECOMPUTE_ONLY` | 26.01 | 241,004,544 |
| `HIERARCHICAL_HEAD` | 23.02 | 242,970,624 |
| `TOPOLOGY_APPLY` | 26.43 | 242,970,624 |

## Acceptance

- `legal_terminal_set_unchanged`: **pass**
- `matched_workload_identical_action_sequences`: **pass**
- `matched_quality_every_step`: **pass**
- `exact_singleton_authority_preserved`: **pass**
- `head_always_defers_off_mode`: **pass**
- `zero_model_forwards_without_head`: **pass**
- `node_pass_improvement_at_matched_quality`: **pass**

Disposition: **execution_sparsity_supported** (evidence tier `fixture_wiring`, systems claim: False).

Fixture/wiring-scale arm comparison (CPU, deterministic small request set). Node-pass counters model the production decode path; the harness re-materializes authoritative trees every step to certify matched quality, so arm wall-clock here is conservative against topology-apply. No wall-clock or systems claim is made; serialized target compression is not reported as inference speed.

Operator legal-set enumeration is budget-capped; truncated operators stay UNKNOWN (never UNSUPPORTED) and forced-singleton detection requires COMPLETE coverage, exactly as in enumerate_operator_legal_set.

The harness re-materializes authoritative trees every step to certify
matched quality, so arm wall-clock is conservative *against*
topology-apply; no wall-clock or systems claim is made, and serialized
target compression is not reported as inference speed. A real systems
claim would require preregistered node-pass/model-call or wall-clock
improvement with exact hardware/workload identity on the neural decode
path, which this fixture does not provide.

The run completed in 75.47s with peak process memory 242,970,624 bytes. AgentV passed 4/4 evidence cases with zero execution errors.

No checkpoint was created, so the model card and README checkpoint summary
do not change.
