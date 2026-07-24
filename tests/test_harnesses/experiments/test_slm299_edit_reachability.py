"""SLM-299 (LAR1-03) edit-reachability analyzer, audit report, and gate tests."""

from __future__ import annotations

import random

import pytest

from slm_training.harnesses.experiments.slm299_edit_reachability import (
    DEFAULT_SEED_SOURCE,
    Verdict,
    add_container_action,
    analyze_reachability,
)
from slm_training.models.tree_edit_diffusion import (
    ACTION_ADD,
    ACTION_REMOVE,
    ACTION_REPLACE,
    Edit,
    TreeEditSpace,
    parse_statements,
    render_statements,
)
from scripts.run_slm299_reachability_audit import (
    build_report,
    render_markdown,
    summarize_suite,
)

SEED = DEFAULT_SEED_SOURCE  # 'root = Stack([], "column")' — the X22 minimal seed


# (a) reachable leaf replacement / addition -------------------------------

def test_reachable_single_leaf_add_has_exact_lower_bound() -> None:
    case = analyze_reachability(
        SEED,
        'root = Stack([cta], "column")\ncta = Button(":cta.label")',
        slot_inventory=[":cta.label"],
    )
    assert case.verdict is Verdict.PROVEN_REACHABLE
    assert case.reason_code == "reached"
    assert case.edit_lower_bound == 1
    assert case.path == [
        {"action": "ADD", "stmt": 0, "comp": "Button", "slot": ":cta.label"}
    ]


def test_reachable_leaf_replacement_from_seeded_state() -> None:
    # REPLACE is kind-preserving: TextContent -> Button on an existing leaf.
    case = analyze_reachability(
        'root = Stack([n0], "column")\nn0 = TextContent(":body")',
        'root = Stack([cta], "column")\ncta = Button(":body")',
        slot_inventory=[":body"],
    )
    assert case.verdict is Verdict.PROVEN_REACHABLE
    assert case.edit_lower_bound == 1
    assert case.path[0]["action"] == "REPLACE"


def test_seed_identity_is_zero_edits() -> None:
    case = analyze_reachability(SEED, SEED, slot_inventory=[":x"])
    assert case.verdict is Verdict.PROVEN_REACHABLE
    assert case.edit_lower_bound == 0


# (b) multi-container target ----------------------------------------------

def test_multi_container_target_needs_container_add() -> None:
    case = analyze_reachability(
        SEED,
        'root = Stack([card], "column")\ntitle = TextContent(":t")\ncard = Card([title], "column")',
        slot_inventory=[":t"],
    )
    assert case.verdict is Verdict.PROVEN_UNREACHABLE
    assert case.reason_code == "needs_container_add"
    assert case.edit_lower_bound is None


# (c) state/query/action (V0.5) program forms ------------------------------

@pytest.mark.parametrize(
    "target",
    [
        'q = Query("getUser")\nroot = Stack([], "column")',
        'a = Action("submit")\nroot = Stack([], "column")',
        'r = State("draft")\nroot = Stack([], "column")',
        '!v0.5\nroot = Stack([], "column")',
    ],
)
def test_pack_feature_targets_are_unsupported(target: str) -> None:
    case = analyze_reachability(SEED, target, slot_inventory=[":x"])
    assert case.verdict is Verdict.PROVEN_UNREACHABLE
    assert case.reason_code == "unsupported_pack_feature"


# (d) UNKNOWN budget cases are never unreachable ---------------------------

_FIVE_LEAF_TARGET = (
    'root = Stack([a, b, c, d, e], "column")\n'
    "a = TextContent(\":x\")\n"
    "b = TextContent(\":x\")\n"
    "c = TextContent(\":x\")\n"
    "d = TextContent(\":x\")\n"
    "e = TextContent(\":x\")"
)


def test_tiny_budget_yields_unknown_not_unreachable() -> None:
    case = analyze_reachability(
        SEED, _FIVE_LEAF_TARGET, slot_inventory=[":x"], max_edits=2
    )
    assert case.verdict is Verdict.UNKNOWN_BUDGET
    assert case.reason_code == "budget"
    # And it is genuinely reachable with enough budget — the UNKNOWN was a
    # budget artifact, never a proof of unreachability.
    full = analyze_reachability(SEED, _FIVE_LEAF_TARGET, slot_inventory=[":x"])
    assert full.verdict is Verdict.PROVEN_REACHABLE
    assert full.edit_lower_bound == 5


def test_unknowns_are_excluded_from_reachable_fraction() -> None:
    cases = [
        ("hit", analyze_reachability(SEED, SEED, slot_inventory=[":x"])),
        (
            "unknown",
            analyze_reachability(
                SEED, _FIVE_LEAF_TARGET, slot_inventory=[":x"], max_edits=2
            ),
        ),
    ]
    summary = summarize_suite(cases)
    assert summary["n_unknown_budget"] == 1
    assert summary["n_decided"] == 1
    # 1/1 decided reachable — the UNKNOWN is not counted as unreachable.
    assert summary["reachable_fraction"] == 1.0


# (e) synthetic ADD_CONTAINER what-if --------------------------------------

_CARD_TARGET = 'root = Stack([card], "column")\ncard = Card([], "column")'


def test_synthetic_add_container_flips_verdict_and_is_marked() -> None:
    real = analyze_reachability(SEED, _CARD_TARGET, slot_inventory=[":x"])
    assert real.verdict is Verdict.PROVEN_UNREACHABLE
    assert real.reason_code == "needs_container_add"

    what_if = analyze_reachability(
        SEED,
        _CARD_TARGET,
        slot_inventory=[":x"],
        extra_actions=[add_container_action()],
    )
    assert what_if.verdict is Verdict.PROVEN_REACHABLE
    assert what_if.details["uses_synthetic_actions"] is True
    assert any(step.get("synthetic") for step in what_if.path)


# (f) transition parity with TreeEditSpace.apply ---------------------------

def _random_statements(rng: random.Random) -> list:
    depth = rng.randint(0, 3)
    statements = parse_statements(SEED)
    assert statements is not None
    space = TreeEditSpace()
    inventory = [":x", ":y"]
    for _ in range(depth):
        step = space.sample_mutation(statements, inventory, rng)
        if step is None:
            break
        statements, _ = step
    return statements


def test_transition_parity_with_tree_edit_space() -> None:
    # The analyzer must enumerate exactly the children TreeEditSpace.apply
    # admits — no more, no fewer — on random reachable states.
    from slm_training.harnesses.experiments.slm299_edit_reachability import (
        _enumerate_children,
    )

    rng = random.Random(20260724)
    inventory = [":x", ":y"]
    space = TreeEditSpace()
    for _ in range(12):
        statements = _random_statements(rng)
        via_analyzer = {
            render_statements(child)
            for child, _ in _enumerate_children(space, statements, inventory)
        }
        via_reference: set[str] = set()
        n_comp = len(space.components)
        for stmt in range(len(statements)):
            for comp in range(n_comp):
                nxt = space.apply(statements, Edit(ACTION_REPLACE, stmt, comp), inventory)
                if nxt is not None:
                    via_reference.add(render_statements(nxt))
                for slot in range(len(inventory)):
                    nxt = space.apply(
                        statements, Edit(ACTION_ADD, stmt, comp, slot), inventory
                    )
                    if nxt is not None:
                        via_reference.add(render_statements(nxt))
            nxt = space.apply(statements, Edit(ACTION_REMOVE, stmt), inventory)
            if nxt is not None:
                via_reference.add(render_statements(nxt))
        assert via_analyzer == via_reference


# (g) ship-gate integration -------------------------------------------------

def _gate_suites() -> dict[str, dict[str, object]]:
    from slm_training.harnesses.model_build.ship_gates import DEFAULT_SHIP_GATES

    suites: dict[str, dict[str, object]] = {}
    for suite, mins in DEFAULT_SHIP_GATES.items():
        suites[suite] = {
            "n": 25,
            "parse_rate": 1.0,
            "syntax_parse_rate": 1.0,
            "placeholder_validity": 1.0,
            "fallback_count": 0,
            **{metric: min(1.0, floor + 0.3) for metric, floor in mins.items()},
        }
    return suites


def test_ship_gates_block_suite_with_partial_reachability() -> None:
    from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates

    result = evaluate_ship_gates(
        _gate_suites(), suite_reachability={"smoke": 0.5}
    )
    assert result["pass"] is False
    assert any(
        failure.startswith("reachability_unproven:smoke")
        for failure in result["measurement_integrity_failures"]
    )
    assert result["gates"]["smoke:reachability_unproven"] is False


def test_ship_gates_pass_at_full_reachability() -> None:
    from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates

    suites = _gate_suites()
    result = evaluate_ship_gates(
        suites, suite_reachability={suite: 1.0 for suite in suites}
    )
    assert result["pass"] is True
    assert not result["measurement_integrity_failures"]


def test_ship_gates_unchanged_when_reachability_map_omitted() -> None:
    from slm_training.harnesses.model_build.ship_gates import evaluate_ship_gates

    suites = _gate_suites()
    baseline = evaluate_ship_gates(suites)
    explicit = evaluate_ship_gates(suites, suite_reachability=None)
    sparse = evaluate_ship_gates(suites, suite_reachability={"not_a_suite": 0.1})
    assert baseline["pass"] is True
    assert explicit["pass"] is True
    assert sparse["pass"] is True
    assert baseline["failures"] == explicit["failures"] == sparse["failures"]


# (h) report determinism ----------------------------------------------------

def _mini_corpora() -> dict[str, list[dict[str, object]]]:
    return {
        "smoke": [
            {
                "id": "hit",
                "openui": 'root = Stack([b], "column")\nb = Button(":x")',
                "placeholders": [":x"],
            },
            {
                "id": "miss",
                "openui": _CARD_TARGET,
                "placeholders": [":x"],
            },
        ],
        "ood": [],
    }


def test_audit_report_is_deterministic() -> None:
    first = build_report(
        _mini_corpora(), max_edits=8, node_budget=200, generated_at="2026-07-24T00:00:00Z"
    )
    second = build_report(
        _mini_corpora(), max_edits=8, node_budget=200, generated_at="2026-07-24T00:00:00Z"
    )
    first.pop("version_stamp")
    second.pop("version_stamp")
    assert first == second
    assert render_markdown(first) == render_markdown(second)
    assert first["suites"]["smoke"]["reachable_fraction"] == 0.5
    assert first["suites"]["ood"]["status"] == "corpus_unavailable"
    assert first["x22_evidence_annotations"]
