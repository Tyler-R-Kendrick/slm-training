# DSH3-08 immutable conversation state graph

SLM-376 replaces hidden mutable edit-session context with a frozen,
content-addressed state/turn graph. Every AST edge is derived from one
successful pack-owned `OperatorApplicationV1`; history operations only select
an existing verified node or create a branch-local duplicate.

## State and turn records

`ConversationStateNodeV1` contains:

- its optional parent state ID;
- a branch digest;
- the full canonical `OperatorStateV1`; and
- the state- and branch-bound `ReferenceTableV1`.

The state ID hashes the parent, branch, exact state/AST digests, and reference
table fingerprint. `TurnArtifactV1` records the operation, input/output state
IDs, compiler/source provenance, and the exact application record/ID for an AST
edit. History turns cannot carry AST applications.

`ConversationTraceV1` is a frozen tuple of nodes and chronological turns plus an
explicit current-state ID. It has no redo stack, mutable branch pointer, or
ambient conversation store.

## AST and history operations

`append_operator_turn`:

1. requires current request-local argument refs;
2. checks source provenance against the current canonical source;
3. replays the successful application through the owning pack/library;
4. validates the caller-supplied output reference table against the exact
   resulting state and current branch; and
5. adds one child node and one `AST_EDIT` turn.

A parent may have only one child on a given branch. Attempting another
same-branch next turn fails with an instruction to fork; any number of explicit
sibling forks may use distinct branch digests.

`UNDO` selects the exact parent. `REDO` requires an explicit direct-child state
ID, so redo behavior is graph-derived rather than stored in a hidden stack.
`CHECKOUT_STATE` selects an existing non-current node. These operations add
turn artifacts but never copy or mutate AST state.

`FORK` creates a new child node with the same canonical state and a branch
digest derived from the root plus an explicit nonce. It deterministically:

- remaps semantic, parent, parent-order, and runtime-symbol fingerprints;
- reallocates opaque references under a branch-derived seed; and
- retains the source table's request ID, compiler facts, and value types.

Old refs therefore fail as missing in the fork table, while using the source
table under the fork digest fails `ref.cross_branch`.

## Replay boundary

`replay_conversation_trace` starts from the pack-authorized root and walks every
turn in order. It verifies:

- cursor-contiguous input/output IDs;
- current source/request/compiler provenance;
- exact parent and branch rules for AST, undo, redo, and fork edges;
- operator application replay and every intermediate canonical state/AST;
- state/reference table state and branch binding;
- deterministic fork reconstruction;
- the final explicit cursor; and
- absence of orphan nodes.

Callers may provide one fixed immutable operator library or an explicit
state-to-authority resolver. The resolver exists for production operator
families whose executor closures are rebuilt from each node's branch-local
compiler context; it is invoked by state ID and its result must replay the
recorded application exactly.

## Evidence and scope

Deterministic unit controls cover multi-edit replay, undo/redo identity,
checkout, fork and branch editing, old-ref invalidation, sibling-fork isolation,
same-branch child refusal, deterministic trace fingerprints, missing history
boundaries, stale application/output-table provenance, corrupted intermediate
state detection, resolver/fixed-library exclusivity, and frozen records.

The explicit edit representation is adapted from
[Yin et al., 2019](https://arxiv.org/abs/1810.13337); systematic
source/follow-up controls are only general motivation from
[Saha and Kanewala, 2018](https://arxiv.org/abs/1802.07361). This state graph
does not implement or evaluate either paper's learned or experimental method.
No data synthesis, train, eval, benchmark, matrix, checkpoint, model-card,
ship-gate, or model-quality claim is produced.
