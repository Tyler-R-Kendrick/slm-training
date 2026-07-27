"""VAR1-01 (SLM-424): hypothetical property-mutation/component-widening probe.

Mirrors ``tests/test_harnesses/experiments/test_slm299_edit_reachability.py``'s
synthetic-what-if pattern: each hypothetical action is proven to (a) never
fire without its capability, (b) re-validate through the real parser, and
(c) mark any path that used it ``uses_synthetic_actions``, so a what-if proof
can never be confused with a real reachability claim.
"""

from __future__ import annotations

from slm_training.harnesses.experiments.slm299_edit_reachability import (
    DEFAULT_SEED_SOURCE,
    Verdict,
    analyze_reachability,
    component_widen_action,
    set_property_action,
)
from slm_training.models.tree_edit_diffusion import TreeEditSpace
from scripts.run_slm299_reachability_audit import build_report
from scripts.run_var1_01_set_property_probe import (
    _categorize_flips,
    _confirmed_and_inconclusive,
    _decision_sentence,
    _extra_actions_for,
    _record_target_components,
)

SEED = DEFAULT_SEED_SOURCE  # 'root = Stack([], "column")'

# Reused verbatim from test_slm299_edit_reachability.py's own _CARD_TARGET --
# a target already proven parseable/valid across every environment this
# suite runs in, so the component_widen tests below never depend on any
# other component's exact grammar shape.
_CARD_TARGET = 'root = Stack([card], "column")\ncard = Card([], "column")'


# (a) set_property hypothetical: root-rest mutation ------------------------


def test_root_direction_change_is_unreachable_at_baseline() -> None:
    case = analyze_reachability(SEED, 'root = Stack([], "row")', slot_inventory=[])
    assert case.verdict is Verdict.PROVEN_UNREACHABLE
    assert case.reason_code == "needs_direction_change"


def test_set_property_flips_root_direction_change_and_is_marked() -> None:
    what_if = analyze_reachability(
        SEED,
        'root = Stack([], "row")',
        slot_inventory=[],
        extra_actions=[set_property_action()],
    )
    assert what_if.verdict is Verdict.PROVEN_REACHABLE
    assert what_if.details["uses_synthetic_actions"] is True
    assert any(step.get("synthetic") for step in what_if.path)
    assert what_if.path[0]["action"] == "SET_PROPERTY"


def test_set_property_capability_gates_the_invariant_precisely() -> None:
    # Without the capability, the invariant proof fires before BFS ever runs
    # (no states_visited recorded); with it, the proof is skipped and BFS
    # actually searches (states_visited is populated) -- the capability,
    # not BFS luck, is what changes the verdict.
    target = 'root = Stack([], "row")'
    baseline = analyze_reachability(SEED, target, slot_inventory=[])
    assert baseline.details.get("states_visited") is None

    what_if = analyze_reachability(
        SEED, target, slot_inventory=[], extra_actions=[set_property_action()]
    )
    assert what_if.details.get("states_visited") is not None


# (b) component_widen hypothetical: bounded to a narrowed live space -------


def test_unsupported_component_is_unreachable_when_pack_owned_tuple_is_narrow(
    monkeypatch,
) -> None:
    """Demonstrates the real, current-state unsupported_component proof: it
    checks CONTAINER_COMPONENTS by name, so narrowing that tuple (exactly
    the kind of pack-vs-variant drift VAR0-03 exists to close) makes an
    otherwise valid target provably unreachable. Uses Card -- whose validity
    in this exact shape is already exercised throughout the existing test
    suite (``_CARD_TARGET`` above) -- rather than depend on any other
    component's exact grammar shape being parseable across environments."""
    import slm_training.harnesses.experiments.slm299_edit_reachability as module

    monkeypatch.setattr(module._shared_space(), "container_components", ("Stack", "Form"))
    case = module.analyze_reachability(SEED, _CARD_TARGET, slot_inventory=[])
    assert case.verdict is Verdict.PROVEN_UNREACHABLE
    assert case.reason_code == "unsupported_component"


def test_component_widen_flips_a_narrowed_container_components_tuple(
    monkeypatch,
) -> None:
    # Card is already a member of the live, grammar-derived
    # TreeEditSpace.components -- but _check_invariants's unsupported_component
    # proof also checks CONTAINER_COMPONENTS by name (an ``or``, so either
    # narrow-tuple absence alone fires it), so the proof is unsound for
    # anything narrowed out of that tuple: a real REPLACE can already reach
    # it once the flawed proof stops pre-empting BFS. This is the exact
    # narrow-vs-full mismatch VAR0-03 exists to close (see the probe report's
    # "analyzer correctness note"); component_widen_action closes it here as
    # a bounded what-if, without touching production code.
    import slm_training.harnesses.experiments.slm299_edit_reachability as module

    monkeypatch.setattr(module._shared_space(), "container_components", ("Stack", "Form"))
    what_if = module.analyze_reachability(
        SEED, _CARD_TARGET, slot_inventory=[], extra_actions=[component_widen_action(["Card"])]
    )
    assert what_if.verdict is Verdict.PROVEN_REACHABLE


def test_component_widen_generate_is_exercised_when_the_live_space_lacks_it(
    monkeypatch,
) -> None:
    """Narrows both CONTAINER_COMPONENTS and the live TreeEditSpace itself,
    so no real action can enumerate Card at all -- only
    ``component_widen_action``'s own ``generate()`` (which assigns the
    hypothetical component directly, bypassing ``space.components``) can
    produce it. This is the one scenario that actually requires the
    synthetic generator rather than merely lifting the invariant gate."""
    import slm_training.harnesses.experiments.slm299_edit_reachability as module

    narrowed = TreeEditSpace(
        components=("TextContent", "Button", "Image", "TextInput", "Stack", "Form")
    )
    narrowed.container_components = ("Stack", "Form")
    monkeypatch.setattr(module, "_SHARED_SPACE", narrowed)

    baseline = module.analyze_reachability(SEED, _CARD_TARGET, slot_inventory=[])
    assert baseline.verdict is Verdict.PROVEN_UNREACHABLE
    assert baseline.reason_code == "unsupported_component"

    what_if = module.analyze_reachability(
        SEED, _CARD_TARGET, slot_inventory=[], extra_actions=[component_widen_action(["Card"])]
    )
    assert what_if.verdict is Verdict.PROVEN_REACHABLE
    assert what_if.details["uses_synthetic_actions"] is True


def test_record_target_components_is_empty_for_grammar_legal_targets() -> None:
    # Every component a valid target can use is already in the live,
    # grammar-derived TreeEditSpace.components -- component_widen therefore
    # never fires in the real suite corpora today (see the probe report).
    record = {
        "openui": _CARD_TARGET,
    }
    assert _record_target_components(record) == ()


def test_record_target_components_is_bounded_not_open_ended() -> None:
    # An invalid/unparseable target contributes no hypothetical components --
    # this is never an open-ended vocabulary widening.
    assert _record_target_components({"openui": "not a program {{{"}) == ()


# (c) refactor regression guard: extra_actions_for=None changes nothing ----


def _mini_corpora() -> dict[str, list[dict[str, object]]]:
    return {
        "smoke": [
            {
                "id": "direction_change",
                "openui": 'root = Stack([], "row")',
                "placeholders": [],
            },
        ],
        "ood": [],
    }


def test_extra_actions_for_omitted_reproduces_prior_behavior_exactly() -> None:
    corpora = _mini_corpora()
    without_param = build_report(
        corpora, max_edits=8, node_budget=40, generated_at="2026-07-25T00:00:00Z",
        mode="extended",
    )
    with_none = build_report(
        corpora, max_edits=8, node_budget=40, generated_at="2026-07-25T00:00:00Z",
        mode="extended", extra_actions_for=None,
    )
    without_param.pop("version_stamp")
    with_none.pop("version_stamp")
    assert without_param == with_none
    assert without_param["suites"]["smoke"]["reachable_fraction"] == 0.0


def test_extra_actions_for_hook_threads_through_build_report() -> None:
    corpora = _mini_corpora()
    baseline = build_report(
        corpora, max_edits=8, node_budget=40, generated_at="2026-07-25T00:00:00Z",
        mode="extended",
    )
    with_set_property = build_report(
        corpora, max_edits=8, node_budget=40, generated_at="2026-07-25T00:00:00Z",
        mode="extended", extra_actions_for=_extra_actions_for("B_set_property"),
    )
    assert baseline["suites"]["smoke"]["reachable_fraction"] == 0.0
    assert with_set_property["suites"]["smoke"]["reachable_fraction"] == 1.0


# (d) flip categorization never confuses inconclusive with confirmed ------


def test_categorize_flips_distinguishes_reachable_from_unknown_budget() -> None:
    comparison = {
        "verdict_flips": [
            {"id": "a", "v1": "PROVEN_UNREACHABLE", "extended": "PROVEN_REACHABLE"},
            {"id": "b", "v1": "PROVEN_UNREACHABLE", "extended": "UNKNOWN_BUDGET"},
            {"id": "c", "v1": "PROVEN_UNREACHABLE", "extended": "UNKNOWN_BUDGET"},
        ]
    }
    kinds = _categorize_flips(comparison)
    assert kinds == {"confirmed_reachable": 1, "became_unknown_budget": 2, "other": 0}


def test_decision_sentence_reports_confirmation_not_raw_flip_count() -> None:
    # A payload where every arm has far more UNKNOWN_BUDGET reclassifications
    # than confirmed-reachable flips must not be summarized as if all flips
    # were confirmations -- this is the exact bug this test guards against.
    payload = {
        "flips_vs_baseline": {
            "B_set_property": {
                "suite": {
                    "flips_by_kind": {
                        "confirmed_reachable": 1,
                        "became_unknown_budget": 36,
                        "other": 0,
                    }
                }
            },
            "C_component_widen": {
                "suite": {
                    "flips_by_kind": {
                        "confirmed_reachable": 0,
                        "became_unknown_budget": 0,
                        "other": 0,
                    }
                }
            },
            "D_both": {
                "suite": {
                    "flips_by_kind": {
                        "confirmed_reachable": 1,
                        "became_unknown_budget": 36,
                        "other": 0,
                    }
                }
            },
        }
    }
    sentence = _decision_sentence(payload)
    assert "confirmed 1 case" in sentence
    assert "36 more case(s) became UNKNOWN_BUDGET" in sentence
    assert "confirmed_reachable" not in sentence  # never leak raw dict keys
    confirmed, inconclusive = _confirmed_and_inconclusive(payload, "B_set_property")
    assert (confirmed, inconclusive) == (1, 36)
