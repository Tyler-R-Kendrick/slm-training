"""Trusted host configuration and MCP transport for document delivery.

No default endpoint, credentials, REST client, or CLI fallback exists. The host
must expose the installed GitHub connector tools with pinned input schemas.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from slm_training.harness_core.activity_contract import contract_digest


class DeliveryWaiting(ValueError):
    """A durable external prerequisite is not satisfied."""


class ConnectorRejected(ValueError):
    """A connector rejection must never select a fallback or smaller payload."""


class ConnectorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    endpoint: str | None = None
    authorization_env: str | None = None
    tool_names: dict[str, str] = Field(default_factory=dict)
    tool_schema_sha256: dict[str, str] = Field(default_factory=dict)
    max_payload_bytes: int = Field(default=200000, gt=0, le=200000)


def check_payload(config, tool, arguments):
    raw = json.dumps(
        {"name": config.tool_names.get(tool, tool), "arguments": arguments},
        ensure_ascii=True,
    ).encode()
    if len(raw) > config.max_payload_bytes:
        raise ConnectorRejected("connector_full_request_exceeds_review_limit")


@asynccontextmanager
async def host_connector(config, *, allowed_tools, timeout_seconds):
    """MCP 1.x Streamable HTTP; endpoint/auth are explicit trusted host inputs."""
    if (
        isinstance(timeout_seconds, bool)
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("connector_timeout_must_be_positive_and_finite")
    headers = _transport_credentials(config)
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        from jsonschema import validate
    except ImportError as error:
        raise DeliveryWaiting("installed_mcp_1x_and_jsonschema_required") from error

    async with asyncio.timeout(timeout_seconds):
        async with streamablehttp_client(
            config.endpoint,
            headers=headers,
            timeout=30,
            sse_read_timeout=30,
        ) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                schemas = await _schemas(session)
                for tool in allowed_tools:
                    schema = schemas.get(config.tool_names.get(tool))
                    if schema is None or contract_digest(
                        schema
                    ) != config.tool_schema_sha256.get(tool):
                        raise DeliveryWaiting(
                            "connector_tool_schema_missing_or_changed"
                        )

                async def call(tool, arguments):
                    if tool not in allowed_tools:
                        raise ValueError("connector_reader_method_not_allowed")
                    name = config.tool_names.get(tool)
                    schema = schemas.get(name)
                    if schema is None or contract_digest(
                        schema
                    ) != config.tool_schema_sha256.get(tool):
                        raise DeliveryWaiting(
                            "connector_tool_schema_missing_or_changed"
                        )
                    validate(arguments, schema)
                    check_payload(config, tool, arguments)
                    response = await session.call_tool(name, arguments)
                    return response.model_dump(mode="json", by_alias=True)

                yield call


async def _schemas(session):
    schemas, cursor, seen = {}, None, set()
    while True:
        page = await session.list_tools(cursor=cursor)
        for tool in page.tools:
            if tool.name in schemas:
                raise ValueError("duplicate_connector_tool")
            schemas[tool.name] = tool.inputSchema
        cursor = page.nextCursor
        if not cursor:
            return schemas
        if cursor in seen:
            raise ValueError("connector_tool_pagination_loop")
        seen.add(cursor)


async def checked_read(connector, tool, **arguments):
    result = await connector(tool, arguments)
    if result.get("isError") or not isinstance(result.get("structuredContent"), dict):
        raise ConnectorRejected("connector_read_rejected")
    return result["structuredContent"]


def _transport_credentials(config):
    if not config.endpoint:
        raise DeliveryWaiting("trusted_github_connector_endpoint_unavailable")
    url = urlsplit(config.endpoint)
    if (
        url.username
        or url.password
        or url.fragment
        or (
            url.scheme != "https"
            and not (
                url.scheme == "http"
                and url.hostname in {"127.0.0.1", "::1", "localhost"}
            )
        )
    ):
        raise ValueError("unsafe_connector_endpoint")
    headers = {}
    if config.authorization_env:
        token = os.environ.get(config.authorization_env)
        if not token:
            raise DeliveryWaiting("trusted_connector_credential_unavailable")
        headers["Authorization"] = "Bearer " + token
    return headers
