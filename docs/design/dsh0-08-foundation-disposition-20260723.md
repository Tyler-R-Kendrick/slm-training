# DSH0-08 G0 foundation disposition (SLM-352)

**Decision:** supported at the `contract_fixture` evidence level. The exact
surface, plan/state-machine, artifact identity, root-family split,
materialization parity, and certificate/lever-control contracts authorize CAP0
data generation and unblock SLM-353. This does not issue `CERT_CAP0`, evaluate
a model, or make a ship claim.

Machine-readable authority:
[`dsh0-08-foundation-disposition-20260723.json`](dsh0-08-foundation-disposition-20260723.json),
disposition SHA
`82fce9dec8e6ede4ff6279f0f2142f96de6c139afcd414789570f7a56e7b4aca`.

## Fail-closed disposition

| Foundation claim | Status | Evidence class |
| --- | --- | --- |
| symbolic surface exactness | supported | contract fixture |
| plan/state-machine enforcement | supported | contract fixture |
| artifact identity | supported | contract fixture |
| root-family split isolation | supported | contract fixture |
| materialization and no-plan parity | supported | contract fixture |
| certificate and lever fail-closed behavior | supported | contract fixture |

`StagedHarnessFoundationDispositionV1` requires every claim exactly once. A
`rejected`, `unknown`, or `invalid` claim requires an explicit blocker and
makes `require_supported()` fail. It also verifies every cited source artifact
against this checkout before returning support. The checked disposition has
six supported claims, eight frozen identities, no open blockers, and next work
item SLM-353.

## Frozen CAP0 identities

| Authority | Version | Frozen identity |
| --- | --- | --- |
| synthesis plan | `synthesis_plan/v1` | `f0778bd88005687242a73626bdf8b4239750bbc9f6d10c06fae92a6240c9daa1` |
| OpenUI pack surface | `openui` | `b37da286bdbcc41c3227f7ec6a379f78c9c62348be682cdbd7544c4feb77cb2d` |
| symbolic surface policy | `symbolic_surface_policy/v1` | `68954d579b07d7f987e5e8151450d573f7e2c8e5d06d64769d885c4718155acb` |
| artifact schema | `capability_artifacts/v1` | `8dab5f0f627eb84b6164f74e24c729ee2f760202f6cbbc3ce629839c8b3ce89c` |
| split policy | `root_family_split/v1` | `7693fa552cacd889ba57e3d0519c2af32c604ff1b90f1d3230bddc69eb77df6c` |
| materialization | `harness.train_data/v17` | `e2617ffde441c9517f97b64f5acb904d20cb60082044dd4a8a680d6b093187e0` |
| progression gates | `capability_gate/v1` | `2e5207431acae264d5c059f2b82b2b6cb6f39e71a34035e9cccc060de18f3495` |
| lever profile | `config.levers/v19` | `d6ab158421b6fdf45e640bedae19195ca483d652b4795b207066a1f1b32b4500` |

Every identity also binds a repository-relative source path and current
SHA-256 in the JSON.

## Audit correction and rerun

The first combined foundation run passed 54 checks and failed one evidence
integrity assertion. Follow-up whole-foundation verification exposed the same
class of problem in another historical citation: DSH0-01/02 had treated the
complete `versions.json` registry and append-only quality matrix byte hashes as
permanent current-source identities. Those files change whenever unrelated
components or result rows advance, even when the cited staged contracts do not.

The correction removes only those volatile aggregate-file hashes from the
historical source lists. Normalized component/frontier identities remain
explicit, all stable implementation/test/contract artifacts remain
byte-verified, and no functional gate or threshold changed. The exact rerun
then passed all 55 foundation checks:

```text
python -m pytest -q \
  tests/test_dsl/test_language_contract.py \
  tests/test_harnesses/test_synthesis_plan.py \
  tests/test_harnesses/test_capability_artifacts.py \
  tests/test_harnesses/train_data/test_artifact_graph.py \
  tests/test_harnesses/train_data/test_staged_materialization.py \
  tests/test_harnesses/test_capability_gates.py
```

The disposition contract and current-artifact verifier passed 10/10 tests and:

```text
python -m scripts.verify_foundation_disposition \
  docs/design/dsh0-08-foundation-disposition-20260723.json --repo-root .
```

No corpus was built, no train/eval/benchmark ran, no checkpoint was created,
and no AgentEvals/AgentV or model-card update is applicable.
