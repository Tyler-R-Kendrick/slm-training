import importlib.util
import subprocess
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / ".agents/skills/hf-cloud-sagemaker-production-defaults/scripts/teardown.py"
)
SPEC = importlib.util.spec_from_file_location("sagemaker_teardown", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
teardown = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(teardown)


def test_delete_command_failure_is_not_reported_as_success(monkeypatch):
    failure = subprocess.CompletedProcess([], 1, "", "AccessDenied")
    monkeypatch.setattr(teardown, "run_aws", lambda _args: failure)

    with pytest.raises(RuntimeError, match="failed to delete endpoint demo"):
        teardown.delete_aws_resource(["sagemaker", "delete-endpoint"], "endpoint demo")


def test_component_discovery_failure_is_not_treated_as_empty(monkeypatch):
    failure = subprocess.CompletedProcess([], 1, "", "AccessDenied")
    monkeypatch.setattr(teardown, "run_aws", lambda _args: failure)

    with pytest.raises(RuntimeError, match="failed to list inference components"):
        teardown.list_inference_components("endpoint", ["--region", "us-east-1"])


def test_malformed_component_list_is_not_treated_as_empty(monkeypatch):
    success = subprocess.CompletedProcess([], 0, "not-json", "")
    monkeypatch.setattr(teardown, "run_aws", lambda _args: success)

    with pytest.raises(RuntimeError, match="invalid JSON"):
        teardown.list_inference_components("endpoint", ["--region", "us-east-1"])


def test_remaining_components_stop_endpoint_deletion(monkeypatch):
    calls = []

    def run_aws(args):
        calls.append(args)
        if args[:2] == ["sagemaker", "describe-endpoint"]:
            return subprocess.CompletedProcess(args, 0, '{"EndpointConfigName":"cfg"}', "")
        if args[:2] == ["sagemaker", "describe-endpoint-config"]:
            return subprocess.CompletedProcess(args, 0, '{"ProductionVariants":[]}', "")
        if args[:2] == ["sagemaker", "list-inference-components"]:
            return subprocess.CompletedProcess(args, 0, '["component"]', "")
        if args[:2] == ["sagemaker", "describe-inference-component"]:
            return subprocess.CompletedProcess(args, 0, '{"Specification":{}}', "")
        if args[:2] in (
            ["cloudwatch", "describe-alarms"],
            ["application-autoscaling", "describe-scaling-policies"],
            ["application-autoscaling", "describe-scalable-targets"],
        ):
            return subprocess.CompletedProcess(args, 0, "[]", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    times = iter((100.0, 1001.0))
    monkeypatch.setattr(teardown, "run_aws", run_aws)
    monkeypatch.setattr(teardown.time, "time", lambda: next(times))
    monkeypatch.setattr(teardown.sys, "argv", ["teardown.py", "endpoint", "us-east-1"])

    assert teardown.main() == 1
    assert not any(args[:2] == ["sagemaker", "delete-endpoint"] for args in calls)
