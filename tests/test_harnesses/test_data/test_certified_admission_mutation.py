"""Mutation tests: every trainer contract at admission must be load-bearing.

Admission and the trainer must apply the *same* record contracts. They did
not: admission ran one check while ``TwoTowerModel.from_records`` ran four, so
29 programs the trainer refuses reached the certified train bucket and every
screening arm exited non-zero. The repair was to apply all four at admission —
but "we call four functions" is not evidence that four checks are working. A
check can be dead (fully shadowed by another), or silently broken: an earlier
draft of this repair imported one of them from the wrong module, so every
record raised ``ImportError``, was recorded as a refusal, and the build
admitted nothing. Counting calls would not have caught either failure.

So each test here removes exactly one contract — the mutant — and asserts the
gate's verdict changes:

* the record that only the removed contract refuses is now **admitted**
  (the check was the only thing catching it, so it is load-bearing), and
* the other three bad records are **still refused** (the removal is surgical;
  the kill is attributable to that one check and not to collateral damage).

A mutant that survives — admission unchanged with the check gone — means the
check is redundant on every input we can construct, and this file would say
so rather than assert a kill it did not observe. All four are killed today.

Finding worth keeping: ``symbol_only`` is shadowed by ``role_safe`` on every
*document*-kind input we could construct. It is only independently observable
at ``target_kind="lexical"``, where role-safety abstains. Since admission
resolves ``record.target_kind or "document"``, a lexical record is guarded by
that contract alone.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.test_data.certified import _assert_certified_role_safe

pytestmark = pytest.mark.mutation

#: (module path, attribute) of each contract admission applies, keyed by the
#: name used below. Patched at these definition sites because
#: ``_assert_certified_role_safe`` imports them at call time.
_CONTRACT_TARGETS: dict[str, tuple[str, str]] = {
    "no_semantic_labels": (
        "slm_training.data.contract",
        "assert_no_template_semantic_labels",
    ),
    "canonical_markers": (
        "slm_training.data.contract",
        "assert_canonical_template_markers",
    ),
    "symbol_only": (
        "slm_training.dsl.language_contract",
        "assert_symbol_only_output",
    ),
    "role_safe": (
        "slm_training.dsl.analysis.templatize",
        "assert_role_safe_output",
    ),
}

#: Same mapping with the module resolved, so the patch lands on the object the
#: call-time import will read rather than on a string pytest re-resolves.
CONTRACTS: dict[str, tuple[ModuleType, str]] = {
    name: (importlib.import_module(module), attribute)
    for name, (module, attribute) in _CONTRACT_TARGETS.items()
}

_CLEAN_UI = 'root = Stack([c1])\nc1 = TextContent([":slot_0"])'


def _record(**overrides: object) -> ExampleRecord:
    fields: dict[str, object] = {
        "id": "mutation-probe",
        "prompt": "Show a card with a title",
        "openui": _CLEAN_UI,
        "placeholders": [":slot_0"],
        "meta": {},
        "accepted_outputs": [],
    }
    fields.update(overrides)
    return ExampleRecord(**fields)  # type: ignore[arg-type]


#: One record per contract, each refused by that contract and *only* that one.
#: Verified by ``test_each_probe_is_refused_by_exactly_its_own_contract``.
PROBES: dict[str, Callable[[], ExampleRecord]] = {
    # A prompt naming semantic roles: template markers must stay opaque.
    "no_semantic_labels": lambda: _record(
        prompt="Semantic roles: title, body. Show a card"
    ),
    # A non-canonical marker identity in the persisted placeholder list.
    "canonical_markers": lambda: _record(placeholders=["{{slot_0}}"]),
    # Free-form text at lexical kind: the one shape role-safety abstains on.
    "symbol_only": lambda: _record(
        openui='"free text"', placeholders=[], target_kind="lexical"
    ),
    # The exact admission miss: a placeholder in a non-content property.
    "role_safe": lambda: _record(
        openui='root = Stack([c1])\nc1 = RadialChart([":slot_0"], [1])'
    ),
}


def _admits(record: ExampleRecord) -> bool:
    """True when the gate lets the record through."""
    try:
        _assert_certified_role_safe(record)
    except Exception:  # noqa: BLE001 — admission refuses by raising, any type
        return False
    return True


def test_a_clean_record_is_admitted() -> None:
    """Without this the mutants below would be killed by a broken fixture."""
    assert _admits(_record())


@pytest.mark.parametrize("name", sorted(PROBES))
def test_each_probe_is_refused_with_every_contract_intact(name: str) -> None:
    assert not _admits(PROBES[name]())


@pytest.mark.parametrize("name", sorted(PROBES))
def test_each_probe_is_refused_by_exactly_its_own_contract(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe must isolate one contract, or the kill below proves nothing.

    Removing every *other* contract must leave the probe refused: whatever
    catches it is the contract it is named for.
    """
    for other, (module, attribute) in CONTRACTS.items():
        if other != name:
            monkeypatch.setattr(module, attribute, lambda *a, **k: None)

    assert not _admits(PROBES[name]())


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_removing_one_contract_admits_a_record_the_trainer_refuses(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation: delete one check, and the gate must go quiet on its probe."""
    module, attribute = CONTRACTS[name]
    monkeypatch.setattr(module, attribute, lambda *a, **k: None)

    assert _admits(PROBES[name]()), (
        f"{attribute} is not load-bearing: admission refuses its probe even "
        "with the contract removed, so either the probe is not isolating or "
        "the check is redundant"
    )


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_removing_one_contract_leaves_the_other_probes_refused(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The removal is surgical: no collateral loosening."""
    module, attribute = CONTRACTS[name]
    monkeypatch.setattr(module, attribute, lambda *a, **k: None)

    for other, build in PROBES.items():
        if other != name:
            assert not _admits(build()), f"removing {attribute} also freed {other}"


def test_a_contract_that_cannot_be_imported_is_never_a_silent_pass() -> None:
    """The failure mode that shipped once: the check raised instead of judging.

    A wrong import made every record raise ``ImportError``. Admission records
    that as a refusal, which is fail-closed and correct — the corpus went to
    zero rather than admitting junk. Pinned so a future "make it robust"
    refactor cannot turn a broken contract into a silent pass.
    """

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise ImportError("cannot import name 'assert_symbol_only_output'")

    module, attribute = CONTRACTS["symbol_only"]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, attribute, _explode)
        assert not _admits(_record())
