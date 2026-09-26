"""Credential setup shares the hard agent budget and fails closed on expiry."""

import pytest

from slm_training.autoresearch.heal import isolated_agent as owner
from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, tree_manifest
from slm_training.autoresearch.heal.repair_contracts import ProviderEndpoint
from tests.test_autoresearch.test_repair_dispatch import repair_request


@pytest.mark.parametrize("endpoint", [None, ProviderEndpoint(port=1234, model="fixture")])
def test_expired_copy_budget_starts_neither_bridge_nor_worker(tmp_path, monkeypatch, endpoint):
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("answer = 0\n")
    value = repair_request.__wrapped__(tmp_path)
    value = value.model_copy(
        update={
            "allowed_paths": ("module.py",),
            "blocker": value.blocker.model_copy(
                update={"source_digest": manifest_digest(tree_manifest(source))}
            ),
            "grant": value.grant.model_copy(update={"provider_endpoint": endpoint}),
        }
    )
    runner = owner.BubblewrapAgentRunner(source=source, attempt_root=tmp_path / "attempts")
    monkeypatch.setattr(runner, "capability", lambda _: None)
    clock = {"now": 100.0}
    monkeypatch.setattr(owner.time, "monotonic", lambda: clock["now"])
    snapshot = owner.private_snapshot

    def expired_copy(*args):
        clock["now"] += value.grant.interrupt_seconds + 1
        return snapshot(*args)

    monkeypatch.setattr(owner, "private_snapshot", expired_copy)
    monkeypatch.setattr(
        owner, "ProviderBridge", lambda *a, **k: pytest.fail("auth started after expiry")
    )
    monkeypatch.setattr(
        owner, "run_isolated", lambda *a, **k: pytest.fail("worker started after expiry")
    )
    result = runner.run(
        value, ("fixture",), inputs={}, progress=lambda: None, cancelled=lambda: False
    )
    assert result.outcome == "timed_out"
    assert result.seconds == value.grant.interrupt_seconds + 1
    assert (runner.attempt_root / value.digest() / "output/proposal.json").exists()
