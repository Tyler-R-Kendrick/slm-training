"""Pinned host Codex RPC transport. Delivery journal retains mutation authority."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core.bounded_process import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS, OwnedProcessTree


class CodexTransportWaiting(ValueError):
    """Unavailable capability or unknown dispatch outcome; never retry writes."""


class CodexAppServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    executable: str
    executable_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    codex_home: str
    state_dir: str
    server: Literal["codex_apps"] = "codex_apps"
    connector_id: Literal["connector_76869538009648d5b282a4bb21c3d157"]
    link_id: str = Field(min_length=1)

    @field_validator("executable", "codex_home", "state_dir")
    @classmethod
    def absolute_path(cls, value):
        if not Path(value).is_absolute():
            raise ValueError("codex_host_path_must_be_absolute")
        return value


def _executable(config):
    path = Path(config.executable).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != config.executable_sha256 or not os.access(path, os.X_OK):
        raise CodexTransportWaiting("codex_executable_pin_changed")
    return str(path)


def _save_thread(path, value):
    fd, name = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


@asynccontextmanager
async def _owned_state(config):
    root = Path(config.state_dir)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    if root.is_symlink():
        raise CodexTransportWaiting("unsafe_codex_state_directory")
    fd = os.open(root / "thread.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CodexTransportWaiting("codex_owned_thread_busy") from error
        yield root / "thread.json"
    finally:
        os.close(fd)


class _Rpc:
    def __init__(self, process):
        self.process = process
        self.sequence = 0
        self.failed = False
        self.lock = asyncio.Lock()

    async def send(self, value):
        self.process.stdin.write(json.dumps(value).encode() + b"\n")
        await self.process.stdin.drain()

    async def call(self, method, params):
        async with self.lock:
            if self.failed:
                raise CodexTransportWaiting("codex_rpc_failed_requires_reconciliation")
            try:
                return await self._call(method, params)
            except (OSError, ValueError, TypeError, KeyError) as error:
                self.failed = True
                if isinstance(error, CodexTransportWaiting):
                    raise
                raise CodexTransportWaiting("codex_rpc_outcome_unknown") from error
            except BaseException:
                self.failed = True
                raise

    async def _call(self, method, params):
        self.sequence += 1
        request_id = self.sequence
        await self.send({"id": request_id, "method": method, "params": params})
        while True:
            raw = await self.process.stdout.readline()
            if not raw:
                raise CodexTransportWaiting("codex_rpc_disconnected_outcome_unknown")
            message = json.loads(raw)
            if not isinstance(message, dict):
                raise CodexTransportWaiting("codex_rpc_invalid_response")
            if "method" in message:
                if "id" in message:
                    await self.send({"id": message["id"], "error": {
                        "code": -32601, "message": "Interactive request refused"}})
                    raise CodexTransportWaiting("codex_interactive_request_required")
                continue
            if message.get("id") != request_id:
                raise CodexTransportWaiting("codex_rpc_response_identity_mismatch")
            if "error" in message:
                raise CodexTransportWaiting("codex_rpc_error_outcome_unknown")
            result = message.get("result")
            if not isinstance(result, dict):
                raise CodexTransportWaiting("codex_rpc_invalid_result")
            return result


async def _thread(rpc, path, binding):
    await rpc.call("initialize", {
        "clientInfo": {"name": "slm_github_connector", "version": "1"},
        "capabilities": {"experimentalApi": True},
    })
    await rpc.send({"method": "initialized"})
    policy = {"sandbox": "read-only", "approvalPolicy": "never"}
    if path.is_symlink():
        raise CodexTransportWaiting("unsafe_codex_thread_state")
    if path.exists():
        state = json.loads(path.read_text())
        if state.get("binding") != binding or not state.get("thread_id"):
            raise CodexTransportWaiting("codex_owned_thread_binding_changed")
        thread_id = state["thread_id"]
        result = await rpc.call("thread/resume", {
            **policy, "threadId": thread_id, "excludeTurns": True})
        if result["thread"]["id"] != thread_id:
            raise CodexTransportWaiting("codex_resumed_thread_identity_changed")
        return thread_id
    result = await rpc.call("thread/start", {
        **policy, "cwd": str(path.parent), "ephemeral": False,
        "baseInstructions": "Dedicated connector transport. No model turns.",
    })
    thread_id = result["thread"]["id"]
    if not isinstance(thread_id, str) or not thread_id:
        raise CodexTransportWaiting("codex_owned_thread_identity_missing")
    _save_thread(path, {"binding": binding, "thread_id": thread_id})
    return thread_id


async def _tools(rpc, thread_id, config):
    tools, seen, cursor = {}, set(), None
    while True:
        page = await rpc.call("mcpServerStatus/list", {
            "threadId": thread_id, "detail": "toolsAndAuthOnly", "cursor": cursor})
        for server in page["data"]:
            if server["name"] == config.server:
                for name, tool in server["tools"].items():
                    if name in tools:
                        raise CodexTransportWaiting("duplicate_codex_tool")
                    tools[name] = tool
        cursor = page.get("nextCursor")
        if not cursor:
            return tools
        if cursor in seen:
            raise CodexTransportWaiting("codex_tool_pagination_loop")
        seen.add(cursor)


def _pinned_tools(config, allowed, tools):
    pinned = {}
    for name in allowed:
        mapped = config.tool_names.get(name, "")
        tool = tools.get(mapped, {})
        meta = tool.get("_meta", {})
        if (not name.startswith("github_") or mapped != "github." + name[7:]
                or tool.get("name") != mapped
                or meta.get("connector_id") != config.codex_app_server.connector_id
                or meta.get("link_id") != config.codex_app_server.link_id
                or contract_digest(tool.get("inputSchema")) != config.tool_schema_sha256.get(name)):
            raise CodexTransportWaiting("connector_tool_schema_or_identity_changed")
        pinned[name] = tool
    return pinned


def _remaining(deadline):
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise CodexTransportWaiting("codex_process_deadline_exhausted")
    return remaining


async def _observe(tree):
    while True:
        tree.refresh()
        await asyncio.sleep(0.01)


async def _stop(process, tree, deadline):
    # Never signal a numeric PGID after its leader may have been reaped.
    # Canonical owner checks each captured PID's start identity before signaling.
    tree.signal(signal.SIGTERM)
    grace_end = min(deadline, asyncio.get_running_loop().time() + KILL_GRACE_SECONDS / 2)
    try:
        while tree.alive() and asyncio.get_running_loop().time() < grace_end:
            await asyncio.sleep(min(0.01, max(0, grace_end - asyncio.get_running_loop().time())))
    finally:
        tree.signal(signal.SIGKILL)
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining > 0:
        await asyncio.wait_for(process.wait(), remaining)


@asynccontextmanager
async def _process(host, executable, deadline):
    env = {k: v for k, v in os.environ.items()
           if k not in {"CODEX_THREAD_ID", "CODEX_SESSION_ID"}}
    env["CODEX_HOME"] = host.codex_home
    async with asyncio.timeout_at(deadline - KILL_GRACE_SECONDS):
        process = await asyncio.create_subprocess_exec(
            executable, "app-server", "--stdio", cwd=host.state_dir, env=env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
            limit=8 * 1024 * 1024)
    tree = OwnedProcessTree(process.pid)
    # An already-exited root must never be adopted later under a recycled PID.
    tree.identities.setdefault(process.pid, "unobserved-exited-root")
    observer = asyncio.create_task(_observe(tree))
    try:
        yield process
    finally:
        observer.cancel()
        try:
            await observer
        except asyncio.CancelledError:
            pass
        finally:
            await _stop(process, tree, deadline)


@asynccontextmanager
async def codex_connector(config, allowed_tools, timeout_seconds):
    # runtime.run bounds the entire adapter, including synchronous setup that
    # cannot be preempted by asyncio. Never reset its elapsed startup budget.
    deadline = asyncio.get_running_loop().time() + min(timeout_seconds, INTERRUPT_AFTER_SECONDS)
    work_deadline = deadline - KILL_GRACE_SECONDS
    _remaining(work_deadline)
    try:
        from jsonschema import validate
    except ImportError as error:
        raise CodexTransportWaiting("installed_jsonschema_required") from error
    host = config.codex_app_server
    try:
        async with asyncio.timeout_at(deadline):
            executable = _executable(host)
            _remaining(work_deadline)
            async with _owned_state(host) as state:
                _remaining(work_deadline)
                async with _process(host, executable, deadline) as process:
                    async with asyncio.timeout_at(work_deadline):
                        _remaining(work_deadline)
                        rpc = _Rpc(process)
                        thread_id = await _thread(rpc, state, contract_digest(config.model_dump(mode="json")))
                        pinned = _pinned_tools(config, allowed_tools, await _tools(rpc, thread_id, host))

                        async def call(tool, arguments):
                            if tool not in pinned:
                                raise ValueError("connector_reader_method_not_allowed")
                            validate(arguments, pinned[tool]["inputSchema"])
                            # One dispatch; existing journal reconciles unknown writes.
                            return await rpc.call("mcpServer/tool/call", {
                                "threadId": thread_id, "server": host.server,
                                "tool": pinned[tool]["name"], "arguments": arguments})

                        yield call
    except (OSError, KeyError, json.JSONDecodeError) as error:
        raise CodexTransportWaiting("codex_transport_outcome_unknown") from error
