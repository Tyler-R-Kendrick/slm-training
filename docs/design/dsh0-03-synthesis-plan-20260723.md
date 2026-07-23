# DSH0-03 synthesis plan and capability state machine (SLM-347)

**Disposition:** adopt `SynthesisPlanV1` for checked-in staged synthesis
configuration. This is a deterministic contract fixture, not a data build,
train/eval/benchmark, checkpoint, capability certificate, or ship claim.

Machine-readable evidence:
[`dsh0-03-synthesis-plan-20260723.json`](dsh0-03-synthesis-plan-20260723.json).

## Plan authority

The plan records the independent capability, supervision, evaluation, and
difficulty axes plus exact identities for the active `DslPack`, symbolic
surface policy, generators, validators, split policy, gate spec, seeds, and
destinations. JSON and YAML parse into the same strict schema and canonical
content hash; unknown keys and non-integral seeds fail closed.

The committed CAP0 fixture pins:

| Authority | Identity |
| --- | --- |
| schema | `synthesis_plan/v1` |
| plan | `dsh0-cap0-fixture` / `beb3e1eb455b2df9509333a4e5e19112e61fa6cf55bcfbb169a2bedf3761476b` |
| pack | `openui` / `b37da286bdbcc41c3227f7ec6a379f78c9c62348be682cdbd7544c4feb77cb2d` |
| surface policy | `symbolic_surface_policy/v1` |
| generator | `pack.corpus_generator` / `harness.train_data/v15` |
| validators | `pack.oracle` / `v15`; `symbolic_surface` / `v1` |
| gate | `gates.ship/openui_ship_gates_v2` |

`SynthesisPlanRegistry` owns only plans. It resolves pack IDs through the
canonical `DslPack` registry and therefore does not create a parallel language
registry.

## Executable transition boundary

```text
CAP0 synthesis (compiler supervision)
  -- verified CERT_CAP0 --> CAP1 synthesis
  -- verified CERT_CAP1 --> CAP2 synthesis (paraphrase/NL supervision)
  -- verified CERT_CAP2 --> distillation or trace-promotion eligibility
```

Missing, mismatched, or unverified certificate references reject the plan.
Distillation requires `SUP_DISTILL`; trace promotion requires `EVAL_TRACE`.
Unknown component IDs, stale versions, missing pack slots, unsupported surface
policy versions, and illegal action/source combinations reject before a
train-data producer is loaded.

## Canonical train-data seam

`build_train_data` accepts an optional `synthesis_plan_path`, exposed as
`--synthesis-plan`. Preflight runs before producer loading and a successful
build would record the plan hash plus exact pack/policy/component identities in
its manifest. The existing no-plan path remains the default, and
`--curriculum` retains its existing independent meaning.

No corpus was built for this contract task, so synthesis feedback, AgentV
output, checkpoint/model-card updates, and ship gates do not apply.

## Validation

Seven focused tests cover checked-in registry loading, JSON/YAML hash
invariance, every capability/promotion transition, stale or unknown
components, unsupported pack slots, strict schema rejection, pre-producer
failure, and the no-plan default. Repository policy, version stamps, Ruff,
compileall, JSON parsing, and changed-test selection remain delivery gates.

## Next disposition

Proceed to the next DSH0 contract using the exact plan identity. A plan grants
eligibility only; it cannot manufacture a missing capability certificate.
