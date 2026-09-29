"""Trusted-controller fixtures for isolated supervisor operation tests."""

from pathlib import Path


def bind_test_controller(monkeypatch, environment):
    from scripts import autotrain_controller_execution as controller
    from slm_training.harness_core import controller_execution as validation

    root = Path(__file__).resolve().parents[2]
    binding = {"cwd": str(root), "source_digest": "a" * 64, "runtime_digest": "b" * 64}
    monkeypatch.setattr(controller, "configured_controller", lambda _: binding)
    monkeypatch.setattr(validation, "validated_controller_environment", lambda _: (root, environment))
    monkeypatch.setattr(controller, "_operation_controller_binding", lambda request: request.get("controller_execution", binding))
    return binding
