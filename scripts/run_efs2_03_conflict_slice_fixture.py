"""EFS2-03 fixture: compare conflict-slice remasking with full remask and suffix rollback.

This script exercises ``slm_training.harnesses.experiments.conflict_slice_repair``
with synthetic injected topology conflicts.  It does not run a real diffusion
model; it proves the conflict-slice schema, repair policies, matched-budget
accounting, and replay traces are wired correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.conflict_slice_repair import (
    CompletenessClass,
    ConflictSliceV1,
    RepairPolicyName,
    TopologyNode,
    compare_repair_policies,
    save_outcomes,
)
from slm_training.versioning import build_version_stamp


OUTPUT_ROOT = Path("outputs/runs/efs2-03-conflict-slice/iter-efs2-03-20260719")
DOCS_JSON = Path("docs/design/iter-efs2-03-conflict-slice-repair-20260719.json")


def _make_tree() -> TopologyNode:
    """Return a synthetic topology tree with protected and certified nodes."""
    # Leaves
    n6 = TopologyNode(node_id=6, node_type="LITERAL", parent_id=3, active=True)
    n7 = TopologyNode(node_id=7, node_type="LITERAL", parent_id=3, active=True)
    n8 = TopologyNode(node_id=8, node_type="LITERAL", parent_id=4, active=True)
    n9 = TopologyNode(node_id=9, node_type="LITERAL", parent_id=5, active=True)
    n10 = TopologyNode(node_id=10, node_type="LITERAL", parent_id=5, active=True)

    # Intermediate nodes
    n3 = TopologyNode(
        node_id=3, node_type="SLOT", parent_id=1, children=(n6, n7), active=True,
        decision_level=2,
    )
    n4 = TopologyNode(
        node_id=4, node_type="SLOT", parent_id=1, children=(n8,), active=True,
        decision_level=2,
    )
    n5 = TopologyNode(
        node_id=5, node_type="SLOT", parent_id=2, children=(n9, n10), active=True,
        decision_level=3,
    )

    # Components
    n2 = TopologyNode(
        node_id=2,
        node_type="COMPONENT",
        parent_id=0,
        children=(n5,),
        active=True,
        decision_level=1,
        protected=True,  # Authored region must not be remasked.
    )
    n1 = TopologyNode(
        node_id=1,
        node_type="COMPONENT",
        parent_id=0,
        children=(n3, n4),
        active=True,
        decision_level=1,
        certified=True,
    )

    # Root
    n0 = TopologyNode(
        node_id=0,
        node_type="ROOT",
        children=(n1, n2),
        active=True,
        decision_level=0,
    )
    return n0


def _slice_fixture(
    conflict_id: str,
    stage: str,
    reason_code: str,
    failing: tuple[int, ...],
    frontier: tuple[int, ...],
    protected: tuple[int, ...],
    completeness: CompletenessClass,
    tree: TopologyNode,
) -> ConflictSliceV1:
    from slm_training.harnesses.experiments.conflict_slice_repair import (
        _tree_fingerprint,
    )

    return ConflictSliceV1(
        conflict_id=conflict_id,
        stage=stage,  # type: ignore[arg-type]
        reason_code=reason_code,
        failing_node_ids=failing,
        dependency_frontier=frontier,
        protected_node_ids=protected,
        completeness_class=completeness,
        original_state_fingerprint=_tree_fingerprint(tree),
        source_provenance="synthetic_fixture",
    )


def _run_fixture() -> dict[str, object]:
    tree = _make_tree()

    fixtures: list[tuple[str, ConflictSliceV1]] = [
        (
            "wrong_production",
            _slice_fixture(
                conflict_id="wrong_production",
                stage="grammar",
                reason_code="wrong_production",
                failing=(4,),
                frontier=(1, 3),
                protected=(2,),
                completeness="EXACT",
                tree=tree,
            ),
        ),
        (
            "dangling_binding",
            _slice_fixture(
                conflict_id="dangling_binding",
                stage="binding",
                reason_code="unresolved_reference",
                failing=(5, 9),
                frontier=(2, 10),
                protected=(2,),
                completeness="SOUND_OVERAPPROX",
                tree=tree,
            ),
        ),
        (
            "heuristic_schema",
            _slice_fixture(
                conflict_id="heuristic_schema",
                stage="schema",
                reason_code="schema_type_mismatch",
                failing=(3, 6),
                frontier=(1, 7),
                protected=(2,),
                completeness="HEURISTIC",
                tree=tree,
            ),
        ),
    ]

    all_outcomes: dict[str, dict[RepairPolicyName, object]] = {}
    raw_outcomes: dict[str, dict[RepairPolicyName, object]] = {}
    for name, slice_ in fixtures:
        outcomes = compare_repair_policies(
            tree,
            slice_,
            seeds=(0, 1, 2),
            budget_forwards=64,
            budget_verifier_calls=16,
        )
        raw_outcomes[name] = outcomes
        all_outcomes[name] = {
            policy: outcome.to_dict() for policy, outcome in outcomes.items()
        }

    summary = {
        "schema_version": "efs2-03-fixture-summary/v1",
        "fixture_count": len(fixtures),
        "policies": list(all_outcomes[next(iter(all_outcomes))].keys()) if all_outcomes else [],
        "outcomes": all_outcomes,
        "version_stamp": build_version_stamp("harness.experiments"),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DOCS_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Persist per-fixture outcome files for replay.
    for name, outcomes in raw_outcomes.items():
        save_outcomes(
            outcomes,  # type: ignore[arg-type]
            OUTPUT_ROOT / f"outcomes_{name}.json",
        )

    return summary


if __name__ == "__main__":
    summary = _run_fixture()
    print(f"EFS2-03 fixture wrote {OUTPUT_ROOT}/summary.json")
    print(f"EFS2-03 fixture wrote {DOCS_JSON}")
    print(
        "fixtures:",
        list(summary["outcomes"].keys()),
    )
