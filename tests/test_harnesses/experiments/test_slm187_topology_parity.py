"""Tests for SLM-187 (FFE1-01) topology solver/runtime parity fixture harness."""

from __future__ import annotations

from slm_training.dsl.solver.topology_adapter import TopologyAdapterConfig
from slm_training.harnesses.experiments.slm187_topology_parity import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    TopologyParityReport,
    TopologyStateV2,
    TopologyTransitionTuple,
    build_fixture_codec,
    build_fixture_trees,
    build_runtime_proposals,
    compare_solver_runtime_domain,
    derive_topology_state_v2,
    render_markdown,
    run_topology_parity_fixture,
)


HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def test_build_fixture_codec_is_torch_free() -> None:
    codec = build_fixture_codec()
    assert codec.bos_id is not None
    assert "=" in codec.production_to_id
    assert "+Stack" in codec.production_to_id


def test_build_fixture_trees_returns_cases() -> None:
    codec = build_fixture_codec()
    trees = build_fixture_trees(codec)
    assert len(trees) >= 8
    ids = {case_id for case_id, _desc, _root in trees}
    assert "doc_root" in ids
    assert "leaf_slot" in ids


def test_derive_topology_state_v2_carries_tree_fingerprint() -> None:
    codec = build_fixture_codec()
    _case_id, _desc, root = build_fixture_trees(codec)[0]
    config = TopologyAdapterConfig(topology_max_nodes=8, topology_max_active=8)
    state = derive_topology_state_v2(root, codec, config)
    assert state.tree_fingerprint
    assert state.state_fingerprint
    assert state.schema == "TopologyStateV2"


def test_topology_state_v2_round_trip() -> None:
    codec = build_fixture_codec()
    _case_id, _desc, root = build_fixture_trees(codec)[0]
    config = TopologyAdapterConfig(topology_max_nodes=8, topology_max_active=8)
    state = derive_topology_state_v2(root, codec, config, slot_inventory=[":a"])
    recovered = TopologyStateV2.from_dict(state.to_dict())
    assert recovered.tree_fingerprint == state.tree_fingerprint
    assert recovered.state_fingerprint == state.state_fingerprint
    assert recovered.slot_inventory == (":a",)


def test_compare_solver_runtime_domain_returns_case() -> None:
    codec = build_fixture_codec()
    _case_id, _desc, root = build_fixture_trees(codec)[0]
    config = TopologyAdapterConfig(topology_max_nodes=8, topology_max_active=8)
    case = compare_solver_runtime_domain(
        root,
        codec,
        config,
        slot_inventory=[":hero.title"],
        case_id="test_doc_root",
        description="test",
    )
    assert case.case_id == "test_doc_root"
    assert case.solver_domain
    assert case.runtime_domain
    # Runtime EXPAND/KEEP must be a subset of the solver domain for parity.
    assert case.runtime_filtered_domain <= case.solver_domain or not case.parity_ok


def test_runtime_proposals_include_delete_and_contract() -> None:
    codec = build_fixture_codec()
    trees = {cid: root for cid, _desc, root in build_fixture_trees(codec)}
    # doc_statement has an active statement child that can be deleted.
    root = trees["doc_statement"]
    config = TopologyAdapterConfig(topology_max_nodes=8, topology_max_active=8)
    choices = {1: 2}  # TopologyAction.DELETE for the active statement child.
    proposals = build_runtime_proposals(
        root,
        codec,
        config,
        slot_inventory=[":hero.title"],
        action_choices=choices,
    )
    delete_tuples = {t for t in proposals if t.action == "DELETE"}
    assert delete_tuples

    # component_list has an inactive expression parent that can contract.
    root = trees["component_list"]
    choices = {2: 3}  # TopologyAction.CONTRACT for the resolved expression.
    proposals = build_runtime_proposals(
        root,
        codec,
        config,
        slot_inventory=[":hero.title"],
        action_choices=choices,
    )
    contract_tuples = {t for t in proposals if t.action == "CONTRACT"}
    assert contract_tuples


def test_run_topology_parity_fixture_produces_report() -> None:
    report = run_topology_parity_fixture(seed=0)
    assert report.matrix_set == MATRIX_SET
    assert report.matrix_version == MATRIX_VERSION
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert report.cases
    assert "harness.experiments.slm187_topology_parity" in report.version_stamp["components"]


def test_report_round_trip() -> None:
    report = run_topology_parity_fixture(seed=0)
    recovered = TopologyParityReport.from_dict(report.to_dict())
    assert recovered.matrix_set == MATRIX_SET
    assert recovered.experiment_id == EXPERIMENT_ID
    assert len(recovered.cases) == len(report.cases)


def test_render_markdown_contains_caveats() -> None:
    report = run_topology_parity_fixture(seed=0)
    md = render_markdown(report)
    assert "SLM-187" in md
    assert "Claim class:" in md
    assert "wiring / fixture only" in md
    assert "No-go for promotion" in md
    assert "Honest caveats" in md


def test_transition_tuple_round_trip() -> None:
    t = TopologyTransitionTuple(
        node_id=1, action="EXPAND", production_id=5, arity=2, slot_id=1
    )
    recovered = TopologyTransitionTuple.from_dict(t.to_dict())
    assert recovered == t
