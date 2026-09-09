# Autonomy source handoff — 2026-09-09

This is the immutable handoff description for the preserved candidate workspace.
It is an inventory and ownership reference, not release authorization.

- Candidate HEAD: `e0eca9f9910244ecc20eb480d363f1852417b599`
- Baseline: `2c4e2df3f9b9bfb6efe13e88108a57ea2ac9f64c`
- Candidate workspace: `outputs/workspaces/autonomy-candidate`
- Inventory entries: 342
- Inventory digest: `37b802af272e150afb7a6c80bdf4ab1f8f55805e50e998e18f887cb3622a8bb3`
- Tracker digest: `b891d232a26554a7cae73a571d6578d7573bde52561ae9340ce24b69caa9630f`

Every listed path is classified by git status and content digest where the file
is present. A receiving worker must recompute the inventory and reject the
handoff if the candidate tree observation changes. Missing fixture inputs remain
capability predicates; they are not synthesized in this manifest.

## Owner dispositions

All A01–A18 packets are recorded as `implemented_unverified` here. A packet
may move to verified only with current producer-to-consumer evidence and exact
test receipts in its Linear issue. This avoids relabeling older-tree evidence.

## Explicit limits

- Virtualenvs, node_modules, credentials, shared Git metadata, and ignored generated outputs are excluded.
- Rootless verifier capability currently reports NETLINK_ROUTE `Operation not permitted`.
- Hypothesis and live configured-agent repair evidence remain unexecuted.
- Training is stopped; this handoff makes no champion, ship, or model-quality claim.
