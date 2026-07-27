"""DSH3-18 CAP2 operator-policy rebase addendum (SLM-393).

Design-only artifact: audits what already exists after the frozen DSH3-13 CAP2
suite (SLM-381) and the terminal DSH3-17 disposition (SLM-385), so the newer
DSH3 M5/M6/M7 "operator-policy rebase" issues (DSH3-19 .. DSH3-33) neither
duplicate an existing callable/schema/harness nor start ahead of an unmet
dependency. This module never mutates the frozen DSH3-13 suite or the DSH3-17
disposition; it only reads their committed identity fields and records a new,
separately versioned inventory/dependency contract over them.

No runtime behavior changes here: this is evidence bookkeeping, mirroring the
shape of ``slm_training.evals.cap2_disposition``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

#: External aliases this addendum treats as already-satisfied prerequisites
#: (frozen suite authority and this addendum issue itself).
_EXTERNAL_ALIASES = frozenset({"DSH3-13", "DSH3-18"})


class InventoryStatus(str, Enum):
    """Where an existing symbol stands relative to a trained, shippable state."""

    PRODUCTION_READY = "production_ready"
    FIXTURE_ONLY = "fixture_only"
    UNTESTED = "untested"
    MISSING = "missing"


@dataclass(frozen=True)
class InventoryItemV1:
    """One audited existing (or missing) symbol with a commit-pinned anchor."""

    symbol_id: str
    status: InventoryStatus
    code_anchor: str
    note: str
    source_digest: str | None
    schema: str = "cap2_rebase_inventory_item/v1"

    def __post_init__(self) -> None:
        if not self.symbol_id or not self.code_anchor or not self.note:
            raise ValueError(
                "inventory item requires symbol_id, code_anchor, and note"
            )
        if self.status is InventoryStatus.MISSING and self.source_digest is not None:
            raise ValueError("a missing symbol cannot carry a source digest")
        if self.status is not InventoryStatus.MISSING and self.source_digest is None:
            raise ValueError("an existing symbol requires a source digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "symbol_id": self.symbol_id,
            "status": self.status.value,
            "code_anchor": self.code_anchor,
            "note": self.note,
            "source_digest": self.source_digest,
        }


class IssueRebaseAction(str, Enum):
    """What DSH3-18 proposes for a pre-rebase (DSH3-14/15/17) issue."""

    UNCHANGED = "unchanged"
    REBASED = "rebased"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class IssueRebaseNoteV1:
    """Update proposal for one pre-rebase issue (DSH3-14, DSH3-15, DSH3-17)."""

    alias: str
    issue_id: str
    action: IssueRebaseAction
    note: str
    superseded_by: str | None = None
    schema: str = "cap2_rebase_issue_note/v1"

    def __post_init__(self) -> None:
        if not self.alias or not self.issue_id or not self.note:
            raise ValueError(
                "issue rebase note requires alias, issue_id, and note"
            )
        if self.action is IssueRebaseAction.SUPERSEDED and not self.superseded_by:
            raise ValueError("a superseded note requires superseded_by")
        if self.action is not IssueRebaseAction.SUPERSEDED and self.superseded_by:
            raise ValueError("only a superseded note may set superseded_by")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alias": self.alias,
            "issue_id": self.issue_id,
            "action": self.action.value,
            "note": self.note,
            "superseded_by": self.superseded_by,
        }


@dataclass(frozen=True)
class DependencyNodeV1:
    """One DSH3 M5/M6/M7 rebase issue's directly-observed Linear dependencies.

    ``blocked_by`` lists only directly-observed Linear ``blockedBy`` edges (or
    the external ``DSH3-13``/``DSH3-18`` prerequisites) -- never a guessed
    transitive closure. ``target_symbols`` names the existing repository
    symbols the issue must reuse rather than duplicate.
    """

    alias: str
    issue_id: str
    status: str
    blocked_by: tuple[str, ...]
    target_symbols: tuple[str, ...]
    ready: bool
    schema: str = "cap2_rebase_dependency_node/v1"

    def __post_init__(self) -> None:
        if not self.alias or not self.issue_id or not self.status:
            raise ValueError(
                "dependency node requires alias, issue_id, and status"
            )
        if len(set(self.blocked_by)) != len(self.blocked_by):
            raise ValueError("dependency node has duplicate blocked_by entries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "alias": self.alias,
            "issue_id": self.issue_id,
            "status": self.status,
            "blocked_by": list(self.blocked_by),
            "target_symbols": list(self.target_symbols),
            "ready": self.ready,
        }


@dataclass(frozen=True)
class Cap2OperatorPolicyRebaseV1:
    """DSH3-18's terminal addendum artifact."""

    pinned_baseline: str
    frozen_suite_identity: Mapping[str, Any]
    reproducibility_note: Mapping[str, Any]
    inventory: tuple[InventoryItemV1, ...]
    issue_notes: tuple[IssueRebaseNoteV1, ...]
    dependency_map: tuple[DependencyNodeV1, ...]
    stop_rule_triggered: bool
    stop_rule_reason: str
    version_stamp: Mapping[str, Any]
    schema: str = "cap2_operator_policy_rebase/v1"

    def __post_init__(self) -> None:
        if not self.pinned_baseline:
            raise ValueError("rebase addendum requires a pinned baseline")
        symbol_ids = [item.symbol_id for item in self.inventory]
        if len(set(symbol_ids)) != len(symbol_ids):
            raise ValueError("duplicate inventory symbol_id")
        note_aliases = [note.alias for note in self.issue_notes]
        if len(set(note_aliases)) != len(note_aliases):
            raise ValueError("duplicate issue rebase note alias")
        dep_aliases = {node.alias for node in self.dependency_map}
        if len(dep_aliases) != len(self.dependency_map):
            raise ValueError("duplicate dependency-map alias")
        known = dep_aliases | _EXTERNAL_ALIASES
        for node in self.dependency_map:
            unknown = [dep for dep in node.blocked_by if dep not in known]
            if unknown:
                raise ValueError(
                    f"dependency map node {node.alias!r} references unknown "
                    f"alias(es) {unknown!r}"
                )
            expected_ready = all(
                dep in _EXTERNAL_ALIASES
                or next(n for n in self.dependency_map if n.alias == dep).status
                in ("done", "in_review")
                for dep in node.blocked_by
            )
            if node.ready != expected_ready:
                raise ValueError(
                    f"dependency map node {node.alias!r} has ready={node.ready} "
                    f"but its blockers imply {expected_ready}"
                )
        if not self.stop_rule_reason:
            raise ValueError("rebase addendum requires a stop-rule reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pinned_baseline": self.pinned_baseline,
            "frozen_suite_identity": dict(self.frozen_suite_identity),
            "reproducibility_note": dict(self.reproducibility_note),
            "inventory": [item.to_dict() for item in self.inventory],
            "issue_notes": [note.to_dict() for note in self.issue_notes],
            "dependency_map": [node.to_dict() for node in self.dependency_map],
            "stop_rule": {
                "triggered": self.stop_rule_triggered,
                "reason": self.stop_rule_reason,
            },
            "version_stamp": dict(self.version_stamp),
        }


def _digest_file(repo_root: Path, relative_path: str) -> str:
    """SHA-256 over a repo-relative file's current bytes (source-digest manifest)."""
    data = (repo_root / relative_path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def _inventory(repo_root: Path) -> tuple[InventoryItemV1, ...]:
    def item(
        symbol_id: str,
        status: InventoryStatus,
        anchor_path: str,
        anchor_suffix: str,
        note: str,
    ) -> InventoryItemV1:
        digest = (
            None
            if status is InventoryStatus.MISSING
            else _digest_file(repo_root, anchor_path)
        )
        return InventoryItemV1(
            symbol_id=symbol_id,
            status=status,
            code_anchor=f"{anchor_path}{anchor_suffix}",
            note=note,
            source_digest=digest,
        )

    return (
        item(
            "models.local_action_head.LocalFlatHead",
            InventoryStatus.FIXTURE_ONLY,
            "src/slm_training/models/local_action_head.py",
            ":180-234",
            "Per-action embeddings are held in a plain python dict "
            "(`self.action_embeddings`), not an `nn.ParameterDict`/`nn.Embedding`. "
            "Verified empirically: `head.parameters()` and `head.state_dict()` "
            "omit these tensors even though autograd tracks their gradients -- "
            "not optimizer-visible or checkpoint-stable. DSH3-21 targets exactly "
            "this gap and is already In Review (PR #897) as of this addendum.",
        ),
        item(
            "models.local_action_head (GlobalMaskedHead|TernaryDigitHead|"
            "TernaryECOCHead|GrammarFactorizedHead|ResidualTritPlaneHead)",
            InventoryStatus.FIXTURE_ONLY,
            "src/slm_training/models/local_action_head.py",
            ":122-645",
            "Real `nn.Linear`/`nn.Embedding` parameters (optimizer- and "
            "checkpoint-visible), but exercised only by unit fixtures; never "
            "wired into a training loop or the CAP2 harness. `_check_forced` "
            "gives every family the single-legal-action singleton bypass.",
        ),
        item(
            "models.legal_edit_scorer.DirectLegalEditPolicy.decide",
            InventoryStatus.PRODUCTION_READY,
            "src/slm_training/models/legal_edit_scorer.py",
            ":194-216",
            "`count == 1` returns a forced decision with zero model calls before "
            "scoring -- the same singleton-bypass invariant as "
            "`local_action_head._check_forced`, already present on this "
            "adjacent legal-edit path.",
        ),
        item(
            "models.legal_edit_batch (FEATURE_NAMES, _stable_scalar)",
            InventoryStatus.PRODUCTION_READY,
            "src/slm_training/models/legal_edit_batch.py",
            ":16-38",
            "Candidate features -- including `successor_fingerprint` -- are "
            "SHA-derived one-dimensional scalars, not typed embeddings. DSH3-23 "
            "targets replacing this; note the current feature vector already "
            "includes `successor_fingerprint` as a plain feature, which DSH3-23 "
            "must isolate rather than assume absent.",
        ),
        item(
            "models.legal_edit_scorer.LegalEditScorer",
            InventoryStatus.FIXTURE_ONLY,
            "src/slm_training/models/legal_edit_scorer.py",
            ":74-146",
            "Trainable `nn.Module` scorer consuming `legal_edit_batch` features; "
            "exercised in unit tests and small harness scripts (SLM-197/198/199/"
            "200), not the CAP2 M5-M7 pipeline.",
        ),
        item(
            "harnesses.experiments.candidate_selector (SLM-127/EFS3-04)",
            InventoryStatus.FIXTURE_ONLY,
            "src/slm_training/harnesses/experiments/candidate_selector.py",
            ":1-6",
            "Module docstring self-declares 'Wiring/fixture harness only'; "
            "built for general candidate selection with calibrated abstention, "
            "never applied to operator legal-action candidates specifically.",
        ),
        item(
            "dsl.operators.collapse (HardNegativeOutcome, replay_collapsed_"
            "instruction, collapse_conversation_trace)",
            InventoryStatus.UNTESTED,
            "src/slm_training/dsl/operators/collapse.py",
            ":37-41,261-329",
            "`CONFLICT`/`DIFFERENT_RESULT` hard negatives and replay-verified "
            "collapse are implemented, but no test file in `tests/` imports "
            "`slm_training.dsl.operators.collapse` -- zero direct test coverage "
            "today. DSH3-24/25/30 must add coverage when building policy rows "
            "from this module.",
        ),
        item(
            "dsl.operators.legal_set (OperatorSupportVerdict.UNKNOWN, "
            "LegalSetCoverage.PARTIAL)",
            InventoryStatus.PRODUCTION_READY,
            "src/slm_training/dsl/operators/legal_set.py",
            ":43,49,226,244,429,456-457,510,518,537",
            "`UNKNOWN`/`PARTIAL` are already typed and used at the legal-set/"
            "data layer. DSH3-25's real gap is the *learning* side (loss "
            "denominator, DEFER semantics) -- not the coverage taxonomy, which "
            "already exists.",
        ),
        item(
            "dsl.operators.merge._conflict_kind (DELETE_MODIFY heuristic)",
            InventoryStatus.PRODUCTION_READY,
            "src/slm_training/dsl/operators/merge.py",
            ":368-398",
            'Line 385 classifies `DELETE_MODIFY` by substring test '
            '(`"remove" in value or "delete" in value`) over operator IDs -- '
            "exactly the name-dependent heuristic DSH3-19 must replace with an "
            "effect-derived classifier. `STALE_REF` conflicts are raised "
            "earlier and independently (lines 543/550/588), so DSH3-19's "
            "priority requirement is already structurally satisfied.",
        ),
        item(
            "models.decode_stats.DecodeStats",
            InventoryStatus.PRODUCTION_READY,
            "src/slm_training/models/decode_stats.py",
            ":12-175",
            "`compiler_lattice_false_hard_eliminations`, "
            "`compiler_lattice_selector_regret`, "
            "`compiler_lattice_invalid_selected_over_valid`, and "
            "`constrained_dead_ends` (with related counters) already exist and "
            "are reusable as DSH3-31 features; no operator-specific "
            "replay-outcome label (illegal/executor-rejection/replay-failure/"
            "semantic-miss/DEFER) exists yet on top of them.",
        ),
        item(
            "evals.cap2_disposition.Cap2CapabilityDispositionV1 (DSH3-17/"
            "SLM-385)",
            InventoryStatus.PRODUCTION_READY,
            "src/slm_training/evals/cap2_disposition.py",
            ":1-304",
            "Terminal disposition already published (CERT_CAP2 not issued, "
            "DSH4 closed). DSH3-33 must publish a *new* disposition instance "
            "over the rebased M5-M7 evidence rather than mutate this one; "
            "historical evidence at "
            "docs/design/dsh3-17-cap2-disposition-20260723/ stays immutable.",
        ),
    )


def _issue_notes() -> tuple[IssueRebaseNoteV1, ...]:
    return (
        IssueRebaseNoteV1(
            alias="DSH3-14",
            issue_id="SLM-382",
            action=IssueRebaseAction.UNCHANGED,
            note=(
                "Serialized-action control (E803) already ran and is Done; "
                "stands unchanged as the reserved-token discrete-action "
                "baseline the new M5-M7 work is measured against."
            ),
        ),
        IssueRebaseNoteV1(
            alias="DSH3-15",
            issue_id="SLM-383",
            action=IssueRebaseAction.REBASED,
            note=(
                "Rebase onto local_action_head.py: build the compiler-masked "
                "hierarchical head as a new `LocalActionHead` family in that "
                "module (reusing `StateContext`/`LocalActionOutput`/"
                "`ActionDecision` and the `_check_forced` singleton bypass) "
                "instead of a parallel scorer stack. Effectively superseded by "
                "the DSH3-21/22/28 track."
            ),
        ),
        IssueRebaseNoteV1(
            alias="DSH3-17",
            issue_id="SLM-385",
            action=IssueRebaseAction.SUPERSEDED,
            note=(
                "Terminal CERT_CAP2-rejected disposition already published "
                "from the pre-rebase E803 evidence "
                "(docs/design/dsh3-17-cap2-disposition-20260723/); DSH3-33 "
                "supersedes it once M5-M7 evidence exists. Prior historical "
                "evidence is unchanged."
            ),
            superseded_by="DSH3-33",
        ),
    )


#: Linear status snapshot at addendum publication time (DSH3-19 .. DSH3-33
#: only; DSH3-13/18 are handled via ``_EXTERNAL_ALIASES``).
_NODE_STATUS: dict[str, str] = {
    "DSH3-19": "backlog",
    "DSH3-20": "backlog",
    "DSH3-21": "in_review",
    "DSH3-22": "backlog",
    "DSH3-23": "backlog",
    "DSH3-24": "backlog",
    "DSH3-25": "backlog",
    "DSH3-26": "backlog",
    "DSH3-27": "backlog",
    "DSH3-28": "backlog",
    "DSH3-29": "backlog",
    "DSH3-30": "backlog",
    "DSH3-31": "backlog",
    "DSH3-32": "backlog",
    "DSH3-33": "backlog",
}
_DONE_LIKE_STATUSES = frozenset({"done", "in_review"})


def _dependency_map() -> tuple[DependencyNodeV1, ...]:
    """DSH3-19 .. DSH3-33: directly-observed Linear ``blockedBy`` edges only."""

    def node(
        alias: str,
        issue_id: str,
        blocked_by: tuple[str, ...],
        target_symbols: tuple[str, ...],
    ) -> DependencyNodeV1:
        ready = all(
            dep in _EXTERNAL_ALIASES
            or _NODE_STATUS.get(dep, "backlog") in _DONE_LIKE_STATUSES
            for dep in blocked_by
        )
        return DependencyNodeV1(
            alias=alias,
            issue_id=issue_id,
            status=_NODE_STATUS[alias],
            blocked_by=blocked_by,
            target_symbols=target_symbols,
            ready=ready,
        )

    return (
        node(
            "DSH3-19", "SLM-394", ("DSH3-18",),
            ("dsl.operators.merge._conflict_kind", "dsl.operators.merge.MergeConflictKind"),
        ),
        node(
            "DSH3-20", "SLM-395", ("DSH3-19",),
            ("dsl.operators.merge", "dsl.operators.references", "dsl.operators.conversation"),
        ),
        node(
            "DSH3-21", "SLM-396", ("DSH3-18",),
            ("models.local_action_head.LocalFlatHead",),
        ),
        node(
            "DSH3-22", "SLM-397", ("DSH3-18", "DSH3-20"),
            (
                "dsl.operators.contracts",
                "dsl.operators.references",
                "dsl.operators.legal_set",
                "models.legal_edit_batch",
            ),
        ),
        node(
            "DSH3-23", "SLM-398", ("DSH3-21", "DSH3-22"),
            (
                "models.legal_edit_batch.FEATURE_NAMES",
                "models.legal_edit_batch._stable_scalar",
                "models.legal_edit_scorer.LegalEditScorer",
            ),
        ),
        node(
            "DSH3-24", "SLM-399", ("DSH3-18", "DSH3-22"),
            ("dsl.operators.collapse", "data.flow.bridge_corpus"),
        ),
        node(
            "DSH3-25", "SLM-400", ("DSH3-22", "DSH3-24"),
            (
                "dsl.operators.legal_set.OperatorSupportVerdict",
                "dsl.operators.legal_set.LegalSetCoverage",
            ),
        ),
        node(
            "DSH3-26", "SLM-401", ("DSH3-24", "DSH3-25"),
            (
                "harnesses.experiments.slm191_termination_matrix",
                "dsl.operators.conversation",
            ),
        ),
        node(
            "DSH3-27", "SLM-402", ("DSH3-22", "DSH3-23", "DSH3-25"),
            ("dsl.operators.references", "dsl.operators.legal_set"),
        ),
        node(
            "DSH3-28", "SLM-403", (
                "DSH3-21", "DSH3-22", "DSH3-23", "DSH3-24",
                "DSH3-25", "DSH3-26", "DSH3-27",
            ),
            (
                "models.local_action_head",
                "models.dynamic_pointer_scorer",
                "harnesses.experiments.cap2_state_local_action",
                "evals.cap2_operator",
            ),
        ),
        node(
            "DSH3-29", "SLM-404", ("DSH3-25", "DSH3-28"),
            ("models.local_action_head.TernaryECOCHead", "dsl.operators.preference"),
        ),
        node(
            "DSH3-30", "SLM-405", ("DSH3-24", "DSH3-28"),
            ("dsl.operators.collapse", "data.flow.bridge_corpus"),
        ),
        node(
            "DSH3-31", "SLM-406", ("DSH3-28",),
            ("models.decode_stats.DecodeStats", "harnesses.experiments.cap2_state_local_action"),
        ),
        node(
            "DSH3-32", "SLM-407", ("DSH3-28",),
            ("models.decode_stats", "dsl.operators.legal_set", "models.tree_edit_diffusion"),
        ),
        node(
            "DSH3-33", "SLM-408", (
                "DSH3-18", "DSH3-19", "DSH3-21", "DSH3-22",
                "DSH3-25", "DSH3-26", "DSH3-27", "DSH3-28", "DSH3-32",
            ),
            ("evals.cap2_disposition.Cap2CapabilityDispositionV1",),
        ),
    )


def build_cap2_operator_policy_rebase(
    *,
    repo_root: Path,
    cap2_report: Mapping[str, Any],
    reproducibility_note: Mapping[str, Any],
    version_stamp: Mapping[str, Any],
) -> Cap2OperatorPolicyRebaseV1:
    """Build the DSH3-18 addendum over the immutable frozen DSH3-13 report.

    Never regenerates or mutates the frozen suite; only reads its committed
    identity fields (adversarial control: "do not treat design prose as code
    evidence" -- every load-bearing statement in the returned artifact traces
    to either this identity or a file digested from ``repo_root``).
    """
    if cap2_report.get("schema") != "cap2_operator_fixture_report/v1":
        raise ValueError("unsupported frozen CAP2 report")
    if cap2_report["policy_scores"]["oracle"]["gate_pass"] is not True:
        raise ValueError("frozen CAP2 oracle contract did not pass")
    suite = cap2_report["suite"]
    frozen_suite_identity = {
        "evidence_id": "SLM-381.cap2_operator_v1",
        "suite_version": suite["suite_version"],
        "suite_hash": suite["suite_hash"],
        "suite_n": len(suite["cases"]),
        "operator_corpus_fingerprint": suite["operator_corpus_fingerprint"],
        "source_records_fingerprint": suite["source_records_fingerprint"],
        "code_commit": cap2_report["version_stamp"]["code_commit"],
    }
    return Cap2OperatorPolicyRebaseV1(
        pinned_baseline="5cd5b8b",
        frozen_suite_identity=frozen_suite_identity,
        reproducibility_note=dict(reproducibility_note),
        inventory=_inventory(repo_root),
        issue_notes=_issue_notes(),
        dependency_map=_dependency_map(),
        stop_rule_triggered=False,
        stop_rule_reason=(
            "HEAD does not contain a trained typed operator policy with "
            "powered CAP2 evidence (DSH3-28/SLM-403 is unstarted); the "
            "portfolio is not obsolete and this addendum proceeds as an "
            "inventory/dependency contract, not a closure."
        ),
        version_stamp=version_stamp,
    )


__all__ = [
    "Cap2OperatorPolicyRebaseV1",
    "DependencyNodeV1",
    "InventoryItemV1",
    "InventoryStatus",
    "IssueRebaseAction",
    "IssueRebaseNoteV1",
    "build_cap2_operator_policy_rebase",
]
