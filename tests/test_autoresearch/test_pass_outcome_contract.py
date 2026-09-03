"""Contract tests: the heal package and the driver must agree on vocabulary.

The heal package writes receipts; the driver's pass classifier reads them and
decides whether a pass made progress. Nothing links the two but string
literals, and a string that stops matching does not raise — it falls through a
``.get`` default. That is not a hypothetical: ``HealOutcome`` carries both
``verify_failed`` and ``postcondition_failed``, the driver mapped only the
first, and so ``postcondition_failed`` — the verdict the data-rebuild playbook
writes when its record-count postcondition does *not* hold, added precisely to
make a no-op data heal visible — scored as the generic ``heal_attempted``.
The check that exists to expose vacuous heals was itself being flattened.

These tests pin each cross-module agreement as a totality claim rather than as
a list of examples, so the next added outcome, action kind or blocker class
fails here instead of silently landing in a default bucket.
"""

from __future__ import annotations

import importlib.util
import sys
import typing
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]


def _driver() -> Any:
    if "run_autotrain_continuous" in sys.modules:
        return sys.modules["run_autotrain_continuous"]
    path = REPO_ROOT / "scripts" / "run_autotrain_continuous.py"
    spec = importlib.util.spec_from_file_location("run_autotrain_continuous", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_autotrain_continuous"] = module
    spec.loader.exec_module(module)
    return module


def _heal_outcomes() -> frozenset[str]:
    from slm_training.autoresearch.heal import schemas

    return frozenset(typing.get_args(schemas.HealOutcome))


# ---------------------------------------------------------------------------
# heal receipts -> driver pass classification
# ---------------------------------------------------------------------------


def test_every_heal_outcome_has_a_declared_pass_classification() -> None:
    """Totality. A missing entry hides itself in the ``heal_attempted`` default."""
    mapped = set(_driver()._PASS_OUTCOME_BY_HEAL_OUTCOME)
    missing = _heal_outcomes() - mapped

    assert not missing, (
        f"heal outcomes with no declared pass classification: {sorted(missing)} — "
        "each would score as the generic 'heal_attempted' without anyone "
        "deciding that it should"
    )


def test_the_pass_map_declares_no_outcome_the_heal_package_cannot_write() -> None:
    """The other direction: a stale key is a rule that can never fire."""
    stale = set(_driver()._PASS_OUTCOME_BY_HEAL_OUTCOME) - _heal_outcomes()

    assert not stale, f"pass map keys that are not HealOutcome members: {sorted(stale)}"


def test_both_failed_proof_obligations_score_as_a_failed_postcondition() -> None:
    """A no-op heal is a counted failure however the verdict was reached.

    ``verify_failed`` comes from the subprocess verify probe and
    ``postcondition_failed`` from an in-process playbook's count check. They
    are the same fact about the pass, and flattening either one into
    ``heal_attempted`` is how a heal that changed nothing reads as work done.
    """
    mapping = _driver()._PASS_OUTCOME_BY_HEAL_OUTCOME

    assert mapping["verify_failed"] == "heal_postcondition_failed"
    assert mapping["postcondition_failed"] == "heal_postcondition_failed"


def test_only_a_verified_heal_scores_as_progress() -> None:
    """Exactly one outcome may read as a successful heal."""
    mapping = _driver()._PASS_OUTCOME_BY_HEAL_OUTCOME
    progress = {
        outcome
        for outcome, pass_outcome in mapping.items()
        if pass_outcome == "verified_heal"
    }

    assert progress == {"healed"}


def test_the_data_rebuild_playbook_writes_a_mapped_outcome() -> None:
    """The producer side of the drift: pin the literal the playbook emits.

    Reading the constant from the driver proves the consumer is total; this
    proves the producer's verdict is one of the outcomes it is total over.
    """
    from slm_training.autoresearch.heal.playbooks import data_rebuild

    source = Path(data_rebuild.__file__).read_text(encoding="utf-8")
    assert 'outcome = "postcondition_failed"' in source
    assert "postcondition_failed" in _driver()._PASS_OUTCOME_BY_HEAL_OUTCOME


# ---------------------------------------------------------------------------
# heal classification -> playbook coverage
# ---------------------------------------------------------------------------


def test_every_blocker_class_a_playbook_handles_is_a_real_class() -> None:
    """A playbook declaring a class nobody produces can never be selected."""
    from slm_training.autoresearch.heal import discovered_playbooks, schemas

    classes = frozenset(typing.get_args(schemas.BlockerClass))
    for playbook in discovered_playbooks():
        unknown = set(playbook.handles) - classes
        assert not unknown, (
            f"{playbook.playbook_id} handles {sorted(unknown)}, which "
            "classify_blocker never returns"
        )


def test_classification_is_total_over_its_own_class_vocabulary() -> None:
    """Every classifier answer is a declared class, for any input."""
    from slm_training.autoresearch.heal import schemas
    from slm_training.autoresearch.heal.classify import classify_blocker

    classes = frozenset(typing.get_args(schemas.BlockerClass))
    probes = [
        ("repair_harness", "harness_failure:control:experiment_failed"),
        ("repair_harness", "npm ci must run: agentv sdk is unavailable"),
        ("rebuild_data", "records_after == records_before"),
        ("repair_formal", "lake build failed"),
        ("stop_campaign", "paid gpu requires user authority"),
        ("", ""),
        ("nonsense-kind", "nonsense reason"),
    ]
    for kind, reason in probes:
        assert classify_blocker(kind, reason) in classes, (kind, reason)


# ---------------------------------------------------------------------------
# park actions -> typed action schema
# ---------------------------------------------------------------------------


def test_the_park_only_emits_action_kinds_the_schema_declares() -> None:
    """The park writes into a validated handoff, so a bad kind is a crash.

    Pinned here rather than left to the writer's validation because the park
    is on the loop's failure path: a crash there replaces a typed park with a
    traceback, which is the state the recovery was undoing.
    """
    from slm_training.autoresearch.schemas import AutotrainActionV1

    kinds = frozenset(
        typing.get_args(AutotrainActionV1.model_fields["kind"].annotation)
    )
    owners = frozenset(
        typing.get_args(AutotrainActionV1.model_fields["owner"].annotation)
    )

    assert {"rebuild_data", "repair_harness", "next_experiment"} <= kinds
    assert {"synthesis-feedback", "improve-openui-harnesses", "autotrain"} <= owners
