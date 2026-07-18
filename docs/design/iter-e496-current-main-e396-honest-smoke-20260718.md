# E496 current-main E396 compatibility audit — 2026-07-18

E496 syncs the complete E396 checkpoint family from
`hf://buckets/TKendrick/OpenUI/checkpoints/e396-balanced-type-head-continuation-r1`
and verifies `last.pt` SHA-256
`feefa0564490bd1db42f79ff710143ad8ed07ab9e4e324f2744a30f8c2f2eee0`.

With the import root pinned to the clean current-main-derived revision
`bccf2355db8fc4487375ad68a95a7f5220dc770a`, checkpoint loading fails in
about three seconds:

```text
checkpoint state mismatch: missing=[]
unexpected=['slot_component_head.bias', 'slot_component_head.weight']
```

No model prediction, metric row, or AgentV result was produced. Earlier
attempts that used the shared virtualenv's editable import path are explicitly
excluded because they loaded another checkout.

This proves the durability half of the checkpoint contract and falsifies the
deployability half. E490 used a long-lived experimental decoder branch whose
slot-component head and later semantic constraints were never reconciled into
`main`.

**Verdict:** E396 is durable but incompatible with current `main`. E490 remains
branch-only diagnostic evidence. Selectively reconcile the generalized decoder
stack before rerunning E396 quality gates.
