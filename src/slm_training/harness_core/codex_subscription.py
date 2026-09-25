"""Host-only Codex login reuse; the worker receives neither tokens nor auth files.

Refresh uses the installed Codex account/read protocol, not a second OAuth client.
The adapter supports Codex's file credential store and the ChatGPT backend only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import signal
import stat
import time
from pathlib import Path


class SubscriptionCancelled(ValueError):
    """Host authentication was revoked before the worker could start."""


_MAX_AUTH_FILE_BYTES = 1024 * 1024


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise SubscriptionCancelled("provider_grant_cancelled")


def _cached(home):
    try:
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        descriptor = os.open(Path(home) / "auth.json", flags)
        try:
            info = os.fstat(descriptor)
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_nlink != 1
                or info.st_uid != os.geteuid()
                or info.st_mode & 0o077
                or info.st_size > _MAX_AUTH_FILE_BYTES
            ):
                raise ValueError("unsafe host subscription file")
            with os.fdopen(descriptor, "rb", closefd=False) as source:
                raw = source.read(_MAX_AUTH_FILE_BYTES + 1)
        finally:
            os.close(descriptor)
        if len(raw) > _MAX_AUTH_FILE_BYTES:
            raise ValueError("oversized host subscription file")
        auth = json.loads(raw.decode("utf-8"))
        if auth.get("auth_mode") != "chatgpt":
            raise ValueError("subscription login required")
        tokens = auth["tokens"]
        token, account = tokens["access_token"], tokens["account_id"]
        if not all(isinstance(value, str) and value and value.isascii()
                   and not any(ord(c) < 33 or ord(c) == 127 for c in value)
                   for value in (token, account)):
            raise ValueError("invalid subscription credentials")
        encoded = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
        expiry = claims["exp"]
        if type(expiry) not in (int, float) or not math.isfinite(expiry):
            raise ValueError("invalid subscription expiry")
        return token, account, expiry
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError):
        raise ValueError("host_codex_subscription_unavailable") from None


async def _rpc(process, request, cancel_event=None):
    _check_cancel(cancel_event)
    process.stdin.write(json.dumps(request).encode() + b"\n")
    await process.stdin.drain()
    while True:
        _check_cancel(cancel_event)
        try:
            line = await asyncio.wait_for(process.stdout.readline(), 0.05)
        except TimeoutError:
            continue
        if not line:
            break
        message = json.loads(line)
        if message.get("id") == request["id"]:
            if "error" in message:
                raise ValueError("host_codex_subscription_refresh_failed")
            return message["result"]
    raise ValueError("host_codex_subscription_refresh_incomplete")


async def _refresh(home, executable, seconds, cancel_event=None):
    _check_cancel(cancel_event)
    env = {name: os.environ[name] for name in ("HOME", "PATH", "SSL_CERT_FILE", "SSL_CERT_DIR")
           if name in os.environ}
    env["CODEX_HOME"] = str(home)
    process = await asyncio.create_subprocess_exec(
        executable,
        "app-server",
        "--stdio",
        "-c",
        'model_provider="openai"',
        "-c",
        "mcp_servers={}",
        cwd=home, env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, start_new_session=True, limit=65536,
    )
    try:
        async with asyncio.timeout(seconds):
            await _rpc(process, {"id": 1, "method": "initialize", "params": {
                "clientInfo": {"name": "slm_repair_host", "version": "1"}}}, cancel_event)
            process.stdin.write(json.dumps({"method": "notifications/initialized"}).encode() + b"\n")
            await process.stdin.drain()
            result = await _rpc(process, {"id": 2, "method": "account/read",
                                         "params": {"refreshToken": True}}, cancel_event)
            if (result.get("account") or {}).get("type") != "chatgpt":
                raise ValueError("host_codex_subscription_refresh_failed")
    finally:
        # No thread is created by account/read. Kill/reap the auth-only process
        # group even if its JSON-RPC response or startup was interrupted.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def subscription_headers(home, executable, deadline, *, cancel_event=None):
    """Validate a credential lifetime covering the entire bounded worker grant."""
    _check_cancel(cancel_event)
    if not math.isfinite(deadline):
        raise ValueError("provider_grant_invalid_deadline")
    if home is None:
        raise ValueError("host_codex_subscription_not_configured")
    token, account, expiry = _cached(home)
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ValueError("provider_grant_expired")
    if expiry <= time.time() + remaining + 60:
        try:
            asyncio.run(_refresh(home, executable, min(30, remaining), cancel_event))
        except SubscriptionCancelled:
            raise
        except (OSError, ValueError, TimeoutError, TypeError, KeyError, AttributeError):
            raise ValueError("host_codex_subscription_refresh_failed") from None
        token, refreshed_account, expiry = _cached(home)
        if refreshed_account != account:
            raise ValueError("host_codex_subscription_account_changed")
    _check_cancel(cancel_event)
    if deadline <= time.monotonic():
        raise ValueError("provider_grant_expired")
    if expiry <= time.time() + (deadline - time.monotonic()) + 30:
        raise ValueError("host_codex_subscription_expired")
    return {"Authorization": "Bearer " + token, "ChatGPT-Account-ID": account,
            "originator": "codex_cli_rs"}
