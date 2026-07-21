# SLM-188 (FFE1-02): edit-algebra reachability, canonical invariance, and transition certificates (slm188-edit-algebra-20260721)

Matrix set: `slm188_edit_algebra`

Version: `ffe1-02-v1`

Status: **fixture**

**Claim class:** wiring / fixture only. No GPU was used, no trainable weights were updated, and no ship-gate claim is made.

## Hypothesis

The canonical statement-level edit algebra (InsertStatement, DeleteStatement, ReplaceProduction, SetArity, InsertChild, DeleteSubtree, BindSlotPointer, BindReferencePointer, SetEnum) can reach every supported fixture target from a structural sketch seed within declared node/depth/edit bounds, and canonical invariance (idempotence, alpha-renaming, slot permutation, independent-edit commutativity) holds for every transition in the fixture domain.

## Falsifier

A bounded fixture planner/search finds a supported canonical target that is unreachable from a sketch seed under the declared edit budget, or a transition whose certificate does not replay to the same canonical target, or an invariance check (canonical idempotence, alpha, slot permutation, commutativity) that fails.

## Summary

- Cases: 2
- Reachable: 2
- Unreachable (complete): 0
- Unknown (budget): 0
- Unsupported pack feature: 0
- Invariance OK: 2
- Disposition: **reachability_holds**

## Reachability cases

| Case | Seed | Target | Result | Path length | Expansions | Frontier max | Replay OK | Invariants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minimal_stack__hero | minimal_stack | hero | reachable | 3 | 3 | 1 | True | C=True A=True S=True |
| minimal_stack__text_only | minimal_stack | text_only | reachable | 1 | 1 | 1 | True | C=True A=True S=True |

## Invariance results

| Case | Idempotent | Alpha | Slot perm | Commutativity |
| --- | --- | --- | --- | --- |
| minimal_stack__hero | True | True | True | True |
| minimal_stack__text_only | True | True | True | True |

## Disposition

**reachability_holds**

Over the bounded fixture domain, supported targets are reachable, replayable, and canonically invariant.

## Go / no-go decision

**No-go for promotion.** This is a wiring fixture. Real bridge coverage over the full train/eval corpus, v0.5 state/query/action targets, and RICO/deeper-tree fixtures requires the standard solver budget and a trained runtime. Do not publish bridges or start flow/direct-policy training until supported-target reachability is ≥95%, every emitted transition replays, and canonical invariance holds.

## Honest caveats

- Fixture-only wiring evidence: no trained model, checkpoint, or GPU run is involved.
- Reachability uses a deterministic sketch→target planner for ordinary records and bounded exact BFS for tiny closed domains; both emit replay-valid certificates.
- The search budget is intentionally small so the harness stays CPU-only; real bridge coverage needs the standard multi-step solver budget.
- v0.5 state/query/action statements, object literals, member access, and operators are represented in the edit-algebra vocabulary but are not exercised by the current OpenUI statement fixtures; they are reported as 'unsupported_pack_feature' when encountered.
- Statement insertion/deletion is performed through the tree-edit space; the topology-solver edit algebra is reused for local node expansion where possible.

## Reproducibility

```bash
python -m scripts.audit_edit_reachability --describe
python -m scripts.audit_edit_reachability --fixtures
```
