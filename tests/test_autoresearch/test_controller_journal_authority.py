"""Journal integrity and environment pointers cannot issue controller authority."""

import shutil

import pytest

from scripts.autotrain_controller_execution import controller_authority_scope, journal_authority
from slm_training.autoresearch.runtime.activity_runtime import ActivityRuntime
from slm_training.autoresearch.storage import CampaignStore


def test_copied_valid_journal_cannot_replace_live_controller_store(tmp_path, monkeypatch):
    journal = CampaignStore("runtime", tmp_path / "original")
    journal.append_event("fixture", detail={"preserved": True})
    copied_root = tmp_path / "copied"
    shutil.copytree(journal.root, copied_root / "runtime")
    copied = CampaignStore("runtime", copied_root)
    assert copied.verify_event_chain() == journal.verify_event_chain()
    monkeypatch.setenv("AUTOTRAIN_CONTROLLER_JOURNAL", str(copied.root))
    with pytest.raises(ValueError, match="requires_live_controller"):
        with journal_authority(copied):
            pytest.fail("copied journal became issuer")
    with ActivityRuntime(journal) as runtime:
        with controller_authority_scope(runtime, None):
            with journal_authority(journal):
                assert runtime.snapshot() == {}
            with pytest.raises(ValueError, match="requires_live_controller"):
                with journal_authority(copied):
                    pytest.fail("foreign store inherited live authority")
    with pytest.raises(ValueError, match="requires_live_controller"):
        with journal_authority(journal):
            pytest.fail("authority survived controller scope")


def test_controller_identity_cannot_be_an_unmarked_none_pin():
    from pathlib import Path
    from slm_training.harness_core.controller_execution import validate_controller

    binding = {"cwd": str(Path(__file__).resolve().parents[2]),
               "source_digest": None, "runtime_digest": "a" * 64}
    with pytest.raises(ValueError, match="nonempty_identity_pins"):
        validate_controller(binding)
    with pytest.raises(ValueError, match="nonempty_identity_pins"):
        validate_controller({**binding, "source_digest": "a" * 64, "runtime_digest": None})


def test_split_controller_requires_pins_before_request_or_source_checks(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.autotrain_controller_execution import configured_controller, operation_controller

    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="split_controller_requires_explicit_pins"):
        configured_controller(SimpleNamespace())
    with pytest.raises(ValueError, match="split_controller_requires_explicit_pins"):
        operation_controller({"cwd": str(tmp_path)})


@pytest.mark.parametrize("kind", ["repair_config", "delivery_config"])
def test_unbound_config_change_cannot_create_fresh_driver_budget(tmp_path, kind):
    from scripts.autotrain_pending import recover_driver_request
    from slm_training.harness_core.activity_contract import ActivitySpec, contract_digest

    original = {"operation": "driver", "predecessor_campaign_id": "original",
                "repair_config": None, "repair_config_digest": None,
                "delivery_config": None, "delivery_config_digest": None}
    journal = CampaignStore("runtime", tmp_path / "loop")
    with ActivityRuntime(journal) as runtime:
        runtime.register(ActivitySpec(activity_id="retained", family="fixture", kind="control",
            source_digest="a" * 64, environment_digest="b" * 64,
            input_digest=contract_digest(original), output_namespace="retained"))
        journal.append_event("operation_repair_requested", experiment_id="retained", detail={"request": original})
        before = runtime.snapshot()
        changed = {**original, kind: str(tmp_path / "config.json"), kind + "_digest": "a" * 64}
        with pytest.raises(ValueError, match="requires_live_host_binding"):
            recover_driver_request(runtime, changed)
        assert runtime.snapshot() == before
        assert recover_driver_request(runtime, original) == original
