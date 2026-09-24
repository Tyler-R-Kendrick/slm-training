"""Pinned schemas and read capabilities are enforced before MCP dispatch."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core.github_connector import (
    ConnectorConfig,
    DeliveryWaiting,
    host_connector,
)


@pytest.fixture
def session(monkeypatch):
    import mcp
    import mcp.client.streamable_http

    schema = {
        "type": "object",
        "properties": {
            "repository_full_name": {"type": "string"},
            "pr_number": {"type": "integer"},
        },
        "required": ["repository_full_name", "pr_number"],
        "additionalProperties": False,
    }
    calls = []

    class Session:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            calls.append("initialize")

        async def list_tools(self, cursor=None):
            return SimpleNamespace(
                tools=[
                    SimpleNamespace(
                        name="connector.github_get_pr_info", inputSchema=schema
                    )
                ],
                nextCursor=None,
            )

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return SimpleNamespace(
                model_dump=lambda **kwargs: {"structuredContent": {"merged": True}}
            )

    @asynccontextmanager
    async def transport(*args, **kwargs):
        yield None, None, None

    monkeypatch.setattr(mcp, "ClientSession", Session)
    monkeypatch.setattr(mcp.client.streamable_http, "streamablehttp_client", transport)
    config = ConnectorConfig(
        endpoint="https://trusted-host.invalid/mcp",
        tool_names={"github_get_pr_info": "connector.github_get_pr_info"},
        tool_schema_sha256={"github_get_pr_info": contract_digest(schema)},
    )
    return config, calls


def call(config, tool="github_get_pr_info", **arguments):
    async def run():
        async with host_connector(
            config, allowed_tools={"github_get_pr_info"}, timeout_seconds=5
        ) as connector:
            return await connector(tool, arguments)

    return asyncio.run(run())


def test_uses_exact_advertised_pinned_schema(session):
    config, calls = session
    assert call(config, repository_full_name="owner/repo", pr_number=1)[
        "structuredContent"
    ]["merged"]
    assert calls[-1] == (
        "connector.github_get_pr_info",
        {"repository_full_name": "owner/repo", "pr_number": 1},
    )


def test_reader_denies_write_even_when_session_has_credentials(session):
    config, calls = session
    with pytest.raises(ValueError, match="method_not_allowed"):
        call(
            config,
            "github_merge_pull_request",
            repository_full_name="owner/repo",
            pr_number=1,
        )
    assert calls == ["initialize"]


def test_schema_drift_fails_before_dispatch(session):
    config, calls = session
    config.tool_schema_sha256["github_get_pr_info"] = "0" * 64
    with pytest.raises(DeliveryWaiting, match="schema_missing_or_changed"):
        call(config, repository_full_name="owner/repo", pr_number=1)
    assert calls == ["initialize"]


def test_schema_rejects_wrong_arguments_before_dispatch(session):
    from jsonschema import ValidationError

    config, calls = session
    with pytest.raises(ValidationError):
        call(config, repository_full_name="owner/repo", pr_number="one")
    assert calls == ["initialize"]


def test_missing_endpoint_is_not_an_invented_transport():
    with pytest.raises(DeliveryWaiting, match="endpoint_unavailable"):
        call(ConnectorConfig())


@pytest.mark.parametrize(
    "endpoint",
    case_values(__file__, "test_untrusted_endpoint_forms_rejected"),
)
def test_untrusted_endpoint_forms_rejected(endpoint):
    with pytest.raises(ValueError, match="unsafe_connector_endpoint"):
        call(ConnectorConfig(endpoint=endpoint))


@pytest.mark.parametrize("seconds", [0, -1, float("inf"), float("nan"), True])
def test_timeout_must_be_explicit_positive_and_finite(seconds):
    async def run():
        async with host_connector(
            ConnectorConfig(), allowed_tools=set(), timeout_seconds=seconds
        ):
            raise AssertionError("unreachable")

    with pytest.raises(ValueError, match="timeout_must_be_positive_and_finite"):
        asyncio.run(run())


@pytest.mark.parametrize("missing", ["mcp", "jsonschema"])
def test_missing_optional_sdk_is_a_capability_wait(monkeypatch, missing):
    import builtins

    original = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == missing:
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    with pytest.raises(
        DeliveryWaiting, match="installed_mcp_1x_and_jsonschema_required"
    ):
        call(ConnectorConfig(endpoint="https://trusted-host.invalid/mcp"))
