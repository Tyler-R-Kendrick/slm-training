"""Martin package metrics must compute the published formulas exactly.

Ca/Ce/I/A/D and the ADP/SDP/SAP checks drive a CI gate, so each is asserted
against a hand-built package tree with a known answer rather than against
whatever the repository happens to look like today.

Contract: ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import pytest

from slm_training.quality import martin
from slm_training.quality.imports import parse_modules, resolve_edges
from slm_training.quality.martin import ComponentMetrics


def _metrics(**overrides) -> ComponentMetrics:
    base = {
        "name": "pkg",
        "modules": 1,
        "lines": 10,
        "classes": 0,
        "abstract_classes": 0,
        "afferent": 0,
        "efferent": 0,
    }
    return ComponentMetrics(**{**base, **overrides})


def _write(root, relative: str, body: str) -> None:
    target = root / "src" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body)


# --- metric formulas -------------------------------------------------------


def test_instability_is_efferent_over_total_coupling() -> None:
    assert _metrics(afferent=1, efferent=3).instability == 0.75


def test_maximally_stable_component_has_zero_instability() -> None:
    assert _metrics(afferent=10, efferent=0).instability == 0.0


def test_isolated_component_is_treated_as_stable() -> None:
    """0/0 is undefined; a component nobody uses and which uses nothing is
    not a coupling risk, so it is reported as stable rather than crashing."""

    assert _metrics().instability == 0.0


def test_abstractness_is_the_abstract_class_ratio() -> None:
    assert _metrics(classes=4, abstract_classes=1).abstractness == 0.25
    assert _metrics(classes=0).abstractness == 0.0


def test_distance_is_measured_from_the_main_sequence() -> None:
    # On the sequence: A + I == 1.
    assert _metrics(classes=1, abstract_classes=1, efferent=0, afferent=1).distance == 0.0
    # Stable and wholly concrete: the far corner.
    assert _metrics(classes=1, afferent=9, efferent=0).distance == 1.0


def test_zone_of_pain_is_stable_and_concrete() -> None:
    assert _metrics(classes=1, afferent=9, efferent=0).zone == "pain"


def test_zone_of_uselessness_is_unstable_and_abstract() -> None:
    assert (
        _metrics(classes=1, abstract_classes=1, afferent=0, efferent=9).zone
        == "uselessness"
    )


def test_balanced_component_sits_on_the_main_sequence() -> None:
    assert _metrics(classes=2, abstract_classes=1, afferent=1, efferent=1).zone == (
        "main-sequence"
    )


# --- ADP -------------------------------------------------------------------


def test_acyclic_graph_reports_no_cycles() -> None:
    assert martin.cycles({"a": {"b"}, "b": {"c"}, "c": set()}) == []


def test_mutual_dependency_is_a_cycle() -> None:
    assert martin.cycles({"a": {"b"}, "b": {"a"}}) == [["a", "b"]]


def test_longer_cycle_is_reported_whole() -> None:
    graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": set()}
    assert martin.cycles(graph) == [["a", "b", "c"]]


def test_self_import_counts_as_a_cycle() -> None:
    assert martin.cycles({"a": {"a"}}) == [["a"]]


def test_disjoint_cycles_are_reported_separately() -> None:
    graph = {"a": {"b"}, "b": {"a"}, "x": {"y"}, "y": {"x"}}
    assert martin.cycles(graph) == [["a", "b"], ["x", "y"]]


def test_deep_chain_does_not_exhaust_recursion() -> None:
    """The traversal is iterative, so depth is bounded by memory, not by
    Python's recursion limit."""

    depth = 3_000
    graph = {f"n{i}": {f"n{i + 1}"} for i in range(depth)}
    graph[f"n{depth}"] = {"n0"}
    found = martin.cycles(graph)
    assert len(found) == 1
    assert len(found[0]) == depth + 1


# --- SDP / SAP -------------------------------------------------------------


def test_depending_on_a_less_stable_component_violates_sdp() -> None:
    stable = _metrics(name="core", afferent=9, efferent=1)
    volatile = _metrics(name="app", afferent=1, efferent=9)
    found = martin.stable_dependency_violations(
        [stable, volatile], {"core": {"app"}, "app": set()}
    )
    assert found == ["core (I=0.10) -> app (I=0.90)"]


def test_depending_toward_stability_satisfies_sdp() -> None:
    stable = _metrics(name="core", afferent=9, efferent=1)
    volatile = _metrics(name="app", afferent=1, efferent=9)
    assert martin.stable_dependency_violations(
        [stable, volatile], {"app": {"core"}, "core": set()}
    ) == []


def test_sap_ignores_a_concrete_component_with_few_dependents() -> None:
    """A concrete leaf is normal; only a widely depended-on concrete component
    is rigid enough to report."""

    lonely = _metrics(name="leaf", classes=1, afferent=1, efferent=0)
    assert martin.stable_abstraction_violations([lonely]) == []


def test_sap_reports_a_widely_used_concrete_component() -> None:
    hub = _metrics(name="hub", classes=1, afferent=20, efferent=0)
    found = martin.stable_abstraction_violations([hub])
    assert len(found) == 1
    assert "zone of pain" in found[0]
    assert "Ca=20" in found[0]


# --- graph construction ----------------------------------------------------


def test_import_graph_resolves_symbols_to_their_module(tmp_path) -> None:
    """``from pkg.mod import Symbol`` is an edge to ``pkg.mod``, not to a
    non-existent ``pkg.mod.Symbol``."""

    _write(tmp_path, "slm_training/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/core.py", "class Thing:\n    pass\n")
    _write(tmp_path, "slm_training/beta/__init__.py", "")
    _write(
        tmp_path,
        "slm_training/beta/use.py",
        "from slm_training.alpha.core import Thing\n",
    )
    edges = resolve_edges(parse_modules(root=tmp_path))
    assert edges["slm_training.beta"] == {"slm_training.alpha"}


def test_third_party_imports_are_not_components(tmp_path) -> None:
    _write(tmp_path, "slm_training/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/core.py", "import numpy\nimport json\n")
    edges = resolve_edges(parse_modules(root=tmp_path))
    assert edges["slm_training.alpha"] == set()


def test_relative_imports_resolve_to_the_sibling_component(tmp_path) -> None:
    _write(tmp_path, "slm_training/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/core.py", "X = 1\n")
    _write(tmp_path, "slm_training/beta/__init__.py", "")
    _write(tmp_path, "slm_training/beta/use.py", "from ..alpha.core import X\n")
    edges = resolve_edges(parse_modules(root=tmp_path))
    assert edges["slm_training.beta"] == {"slm_training.alpha"}


@pytest.mark.parametrize(
    "body",
    [
        "from abc import ABC\nclass A(ABC):\n    pass\n",
        "from typing import Protocol\nclass A(Protocol):\n    pass\n",
        "import abc\nclass A:\n    @abc.abstractmethod\n    def f(self): ...\n",
    ],
)
def test_abstract_classes_are_counted(tmp_path, body: str) -> None:
    _write(tmp_path, "slm_training/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/__init__.py", body)
    modules = parse_modules(root=tmp_path)
    alpha = next(item for item in modules if item.dotted == "slm_training.alpha")
    assert alpha.classes == 1
    assert alpha.abstract_classes == 1


def test_concrete_class_is_not_counted_as_abstract(tmp_path) -> None:
    _write(tmp_path, "slm_training/__init__.py", "")
    _write(tmp_path, "slm_training/alpha/__init__.py", "class A:\n    pass\n")
    modules = parse_modules(root=tmp_path)
    alpha = next(item for item in modules if item.dotted == "slm_training.alpha")
    assert (alpha.classes, alpha.abstract_classes) == (1, 0)


def test_unparseable_module_is_skipped_not_fatal(tmp_path) -> None:
    _write(tmp_path, "slm_training/__init__.py", "")
    _write(tmp_path, "slm_training/broken.py", "def (:\n")
    assert [item.dotted for item in parse_modules(root=tmp_path)] == ["slm_training"]
