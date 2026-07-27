# SLM-395 post-merge continuation audit

Status: fixture wiring evidence passed; not a model or ship claim.

The audit creates two branch-local reference clones and applies exact
fixture edits to find one supported disjoint merge locally, then rebuilds the merged
reference table through the canonical topology context. It enumerates a
fresh legal set, appends one follow-up turn, and replays that trace.

## Exact local result

- branch candidates considered: `1`
- selected merge: `93e4fd68e05ef2f882190afcad86f43f8bc3d02103a5b5fd3553144955fad8f4`
- fresh table fingerprint: `ed41afee995dbd14facf85104d23ddf53d7a0520788f478104c43a163edad041`
- continuation trace turns: `1`
- legal-set coverage: `partial`

Old branch tables are explicitly refused as stale or cross-branch. A
different allocation seed preserves semantic membership while changing
the opaque table fingerprint. Legacy `branch_merge/v1` artifacts remain
terminal: only `branch_merge_continuation/v1` can start a new trace.

This fixture proves wiring and replay only; it is not a training,
evaluation, checkpoint, or ship-grade result.
