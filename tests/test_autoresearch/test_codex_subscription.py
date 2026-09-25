"""Host auth remains private while the existing isolated transport carries requests."""

import base64
import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from slm_training.autoresearch.heal.repair_contracts import ProviderEndpoint
from slm_training.harness_core import codex_subscription as auth
from slm_training.harness_core import provider_bridge as bridge
from tests.test_autoresearch.test_provider_bridge import request


def credentials(home, *, expiry=None, account="fixture-account"):
    claim = {"exp": time.time() + 3600 if expiry is None else expiry}
    payload = base64.urlsafe_b64encode(json.dumps(claim).encode()).decode().rstrip("=")
    token = "fixture." + payload + ".signature"
    path = home / "auth.json"
    path.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
        "access_token": token, "account_id": account, "refresh_token": "never-export-me"}}))
    path.chmod(0o600)
    return token


def test_subscription_refresh_uses_native_host_auth_and_pins_account(tmp_path, monkeypatch):
    credentials(tmp_path, expiry=0)
    calls = []

    async def refresh(home, executable, seconds, cancel_event=None):
        calls.append((home, executable, seconds))
        credentials(home)

    monkeypatch.setattr(auth, "_refresh", refresh)
    headers = auth.subscription_headers(tmp_path, "codex", time.monotonic() + 60)
    assert headers["ChatGPT-Account-ID"] == "fixture-account"
    assert headers["Authorization"].startswith("Bearer fixture.")
    assert len(calls) == 1 and calls[0][2] <= 30
    assert "never-export-me" not in repr(headers)

    async def changed(home, *_args):
        credentials(home, account="another-account")

    credentials(tmp_path, expiry=0)
    monkeypatch.setattr(auth, "_refresh", changed)
    with pytest.raises(ValueError, match="account_changed"):
        auth.subscription_headers(tmp_path, "codex", time.monotonic() + 60)


def test_subscription_never_falls_back_to_api_auth_or_invalid_token(tmp_path):
    (tmp_path / "auth.json").write_text('{"auth_mode":"apikey","OPENAI_API_KEY":"secret"}')
    with pytest.raises(ValueError, match="host_codex_subscription_unavailable"):
        auth.subscription_headers(tmp_path, "codex", time.monotonic() + 60)
    for expiry in (float("nan"), float("inf"), True):
        credentials(tmp_path, expiry=expiry)
        with pytest.raises(ValueError, match="host_codex_subscription_unavailable"):
            auth.subscription_headers(tmp_path, "codex", time.monotonic() + 60)


@pytest.mark.parametrize("unsafe", ["symlink", "hardlink", "permissions", "owner", "fifo"])
def test_subscription_rejects_unsafe_host_auth_files(tmp_path, monkeypatch, unsafe):
    import os

    credentials(tmp_path)
    path = tmp_path / "auth.json"
    if unsafe == "symlink":
        target = tmp_path / "private-auth.json"
        path.replace(target)
        path.symlink_to(target)
    elif unsafe == "hardlink":
        os.link(path, tmp_path / "auth-alias.json")
    elif unsafe == "permissions":
        path.chmod(0o640)
    elif unsafe == "owner":
        owner = path.stat().st_uid
        monkeypatch.setattr(auth.os, "geteuid", lambda: owner + 1)
    else:
        path.unlink()
        os.mkfifo(path)

    with pytest.raises(ValueError, match="^host_codex_subscription_unavailable$"):
        auth.subscription_headers(tmp_path, "codex", time.monotonic() + 60)


def test_subscription_bridge_only_attaches_host_credentials_upstream(tmp_path, monkeypatch):
    token = credentials(tmp_path)
    seen = []

    class Connection:
        sock = None

        def __init__(self, host, *, timeout):
            assert host == "chatgpt.com" and 0 < timeout <= 5

        def connect(self):
            pass

        def request(self, method, path, *, body, headers):
            seen.append((method, path, body, headers))

        def getresponse(self):
            return self

        status = 200

        def getheader(self, _name, default):
            return default

        def read1(self, _size):
            return b""

        def close(self):
            pass

    monkeypatch.setattr(bridge.http.client, "HTTPSConnection", Connection)
    endpoint = ProviderEndpoint(model="fixture", authentication="codex_subscription")
    with bridge.ProviderBridge(endpoint, 5, threading.Event(), subscription=(tmp_path, "codex")) as path:
        status, body, headers = request(path, extra=b"Authorization: attacker\r\nChatGPT-Account-ID: attacker\r\n")
        assert status == 200 and token.encode() not in body and token not in repr(headers)
        assert set(Path(path).parent.iterdir()) == {path}
    assert len(seen) == 1 and seen[0][:2] == ("POST", "/backend-api/codex/responses")
    assert seen[0][3]["Authorization"] == "Bearer " + token
    assert seen[0][3]["ChatGPT-Account-ID"] == "fixture-account"
    assert "never-export-me" not in repr(seen)


def test_subscription_grant_does_not_allow_another_route_or_destination():
    with pytest.raises(ValueError, match="loopback port"):
        ProviderEndpoint(model="fixture", authentication="codex_subscription", port=1234)
    with pytest.raises(ValueError, match="only POST"):
        ProviderEndpoint(model="fixture", authentication="codex_subscription", paths=("/v1/models",))
    with pytest.raises(ValueError, match="loopback port"):
        ProviderEndpoint(model="fixture")
    assert "authentication" not in ProviderEndpoint(model="fixture", port=1234).model_dump()


def test_native_refresh_protocol_requests_no_exported_token(tmp_path):
    executable = tmp_path / "codex"
    executable.write_text(
        "#!/usr/bin/python3\nimport json,os,sys\nfrom pathlib import Path\n"
        "assert sys.argv[1:3]==['app-server','--stdio']\n"
        "assert sys.argv[sys.argv.index('-c')+1]=='model_provider=\"openai\"'\n"
        "assert 'mcp_servers={}' in sys.argv\n"
        "for line in sys.stdin:\n"
        " request=json.loads(line)\n"
        " if request['method']=='notifications/initialized':\n"
        "  Path(os.environ['CODEX_HOME'],'initialized').write_text('yes')\n"
        "  continue\n"
        " if request['method']=='account/read':\n"
        "  assert Path(os.environ['CODEX_HOME'],'initialized').read_text()=='yes'\n"
        "  assert request['params']=={'refreshToken':True}\n"
        "  Path(os.environ['CODEX_HOME'],'refreshed').write_text('account/read')\n"
        " result={} if request['id']==1 else {'account':{'type':'chatgpt'}}\n"
        " print(json.dumps({'id':request['id'],'result':result}),flush=True)\n"
    )
    executable.chmod(0o700)
    asyncio.run(auth._refresh(tmp_path, str(executable), 2))
    assert (tmp_path / "initialized").read_text() == "yes"
    assert (tmp_path / "refreshed").read_text() == "account/read"


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_native_refresh_is_revocable_and_reaps_host_process(tmp_path, stop):
    import os
    executable = tmp_path / "codex"
    executable.write_text(
        "#!/usr/bin/python3\nimport os,time\nfrom pathlib import Path\n"
        "Path(os.environ['CODEX_HOME'],'pid').write_text(str(os.getpid()))\n"
        "time.sleep(30)\n"
    )
    executable.chmod(0o700)
    cancelled = threading.Event()

    def revoke():
        until = time.monotonic() + 2
        while not (tmp_path / "pid").exists() and time.monotonic() < until:
            time.sleep(0.01)
        if stop == "cancel":
            cancelled.set()

    watcher = threading.Thread(target=revoke)
    watcher.start()
    started = time.monotonic()
    error = auth.SubscriptionCancelled if stop == "cancel" else TimeoutError
    try:
        with pytest.raises(error):
            asyncio.run(auth._refresh(tmp_path, str(executable), 2 if stop == "cancel" else 0.5, cancelled))
    finally:
        watcher.join(3)
    assert time.monotonic() - started < 2
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid").read_text()), 0)


@pytest.mark.parametrize("exposure", ["source", "runtime", "auth_link"])
def test_host_credentials_cannot_be_mounted_into_worker(tmp_path, exposure):
    from types import SimpleNamespace
    from slm_training.autoresearch.heal.isolated_agent import BubblewrapAgentRunner

    source, runtime, home = (tmp_path / name for name in ("source", "runtime", "login"))
    for path in (source, runtime, home):
        path.mkdir()
    exposed = source if exposure == "source" else runtime
    if exposure == "auth_link":
        credentials(runtime)
        (home / "auth.json").symlink_to(runtime / "auth.json")
    else:
        home = exposed / "custom-login"
        home.mkdir()
        credentials(home)
    runner = BubblewrapAgentRunner(source=source, attempt_root=tmp_path / "attempts",
                                  runtime_roots=(runtime,), codex_subscription_home=str(home))
    assert runner._credential_mount_failure() == "host_codex_subscription_credentials_exposed_to_worker"
    # Check the real capability path, before any worker/credential bridge starts.
    import unittest.mock
    with unittest.mock.patch("slm_training.autoresearch.heal.isolated_agent.probe_isolation",
                             return_value=SimpleNamespace(available=True)):
        grant = SimpleNamespace(network="approved_provider_only", executable=str(runtime / "codex"),
                                provider_endpoint=ProviderEndpoint(model="fixture", authentication="codex_subscription"))
        assert runner.capability(SimpleNamespace(grant=grant)) == runner._credential_mount_failure()
    assert not (tmp_path / "attempts").exists()


def test_cancelled_or_expired_bridge_never_starts_auth(tmp_path, monkeypatch):
    endpoint = ProviderEndpoint(model="fixture", authentication="codex_subscription")
    monkeypatch.setattr(auth, "subscription_headers", lambda *a, **k: pytest.fail("credentials opened"))
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(auth.SubscriptionCancelled):
        with bridge.ProviderBridge(endpoint, 5, cancelled, subscription=(tmp_path, "codex")):
            pytest.fail("worker started")
    for seconds in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="invalid_duration"):
            bridge.ProviderBridge(endpoint, seconds, threading.Event())


def test_auth_shapes_are_sanitized_and_expiry_rechecked_after_refresh(tmp_path, monkeypatch):
    for content in ("null", "[]", '{"auth_mode":"chatgpt","tokens":null}'):
        (tmp_path / "auth.json").write_text(content)
        with pytest.raises(ValueError, match="^host_codex_subscription_unavailable$"):
            auth.subscription_headers(tmp_path, "codex", time.monotonic() + 2)
    credentials(tmp_path, expiry=0)
    clock = {"now": 100}
    monkeypatch.setattr(auth.time, "monotonic", lambda: clock["now"])

    async def refresh(home, *_args):
        credentials(home)
        clock["now"] = 103

    monkeypatch.setattr(auth, "_refresh", refresh)
    with pytest.raises(ValueError, match="^provider_grant_expired$"):
        auth.subscription_headers(tmp_path, "codex", 102)


def test_runner_charges_auth_setup_and_preserves_cancelled_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from slm_training.autoresearch.heal import isolated_agent as runner_module
    from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, tree_manifest
    from tests.test_autoresearch.test_repair_dispatch import repair_request

    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("answer = 0\n")
    request_value = repair_request.__wrapped__(tmp_path)
    request_value = request_value.model_copy(update={
        "allowed_paths": ("module.py",),
        "blocker": request_value.blocker.model_copy(update={"source_digest": manifest_digest(tree_manifest(source))}),
        "grant": request_value.grant.model_copy(update={"provider_endpoint": ProviderEndpoint(port=1234, model="fixture")})})
    runner = runner_module.BubblewrapAgentRunner(source=source, attempt_root=tmp_path / "attempts")
    monkeypatch.setattr(runner, "capability", lambda _: None)
    monkeypatch.setattr(runner, "_argv", lambda argv: argv)
    clock, cancel = {"now": 100.0}, {"value": False}
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: clock["now"])
    snapshot = runner_module.private_snapshot

    def delayed_snapshot(*args):
        clock["now"] += 1
        return snapshot(*args)

    monkeypatch.setattr(runner_module, "private_snapshot", delayed_snapshot)

    class DelayedBridge:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            clock["now"] += 2
            if cancel["value"]:
                raise auth.SubscriptionCancelled("provider_grant_cancelled")
            return None

        def __exit__(self, *args):
            pass

    def process(spec, *args, **kwargs):
        assert spec.timeout_seconds <= request_value.grant.interrupt_seconds - 3
        clock["now"] += 1
        return SimpleNamespace(outcome=SimpleNamespace(value="completed"), returncode=0, duration_seconds=1,
                               stdout="", stderr="", stdout_truncated=False, stderr_truncated=False)

    monkeypatch.setattr(runner_module, "ProviderBridge", DelayedBridge)
    monkeypatch.setattr(runner_module, "run_isolated", process)
    first = runner.run(request_value, ("fixture",), inputs={}, progress=lambda: None, cancelled=lambda: False)
    assert first.outcome == "completed" and first.seconds == 4
    cancel["value"] = True
    other = request_value.model_copy(update={"attempt_id": "second"})
    second = runner.run(other, ("fixture",), inputs={}, progress=lambda: None, cancelled=lambda: False)
    assert second.outcome == "cancelled" and second.seconds == 3
    assert (runner.attempt_root / other.digest() / "output/proposal.json").exists()


def test_subscription_grant_roundtrip_keeps_auth_home_host_only(tmp_path):
    from slm_training.autoresearch.heal.recovery_dispatch import RecoveryConfig
    from tests.test_autoresearch.test_repair_dispatch import repair_request

    original = repair_request.__wrapped__(tmp_path)
    endpoint = ProviderEndpoint(model="codex5.6sol", authentication="codex_subscription")
    grant = original.grant.model_copy(update={"network": "approved_provider_only", "provider_endpoint": endpoint})
    config = RecoveryConfig(grant=grant, verifier_release="a" * 64, codex_subscription_home=str(tmp_path / "host-auth"))
    restored = RecoveryConfig.model_validate_json(config.model_dump_json())
    assert restored == config and restored.grant.digest() == grant.digest()
    assert "host-auth" not in grant.model_dump_json()
    assert restored.grant.provider_endpoint.authentication == "codex_subscription"
    assert restored.grant.provider_endpoint.model == "codex5.6sol"
    legacy = RecoveryConfig(grant=original.grant, verifier_release="a" * 64)
    serialized = legacy.model_dump(mode="json", exclude={"source_verification_grant", "codex_subscription_home"})
    import hashlib
    from slm_training.lineage.records import canonical_json
    serialized["grant"].pop("expires_at", None)
    serialized.pop("operation_recipes", None)
    assert hashlib.sha256(canonical_json(serialized).encode()).hexdigest() == legacy.proposal_config_digest()


@pytest.mark.parametrize("revoke", ["cancel", "deadline"])
def test_connect_completion_cannot_send_auth_after_grant_revoked(tmp_path, monkeypatch, revoke):
    import http.client
    from unittest.mock import Mock

    credentials(tmp_path)
    cancelled, checked = threading.Event(), threading.Event()
    endpoint = ProviderEndpoint(model="fixture", authentication="codex_subscription")
    owner = bridge.ProviderBridge(endpoint, 5, cancelled, subscription=(tmp_path, "codex"))
    requests = []
    connection = Mock()

    class Connection:
        sock = connection

        def __init__(self, *args, **kwargs):
            pass

        def connect(self):
            pass

        def request(self, *args, **kwargs):
            requests.append(True)

        def close(self):
            checked.set()

    monkeypatch.setattr(bridge.http.client, "HTTPSConnection", Connection)
    with owner as path:
        track = owner.server.track

        def revoke_after_track(sock):
            track(sock)
            if sock is connection:
                if revoke == "cancel":
                    cancelled.set()
                else:
                    owner.server.deadline = time.monotonic() - 1

        monkeypatch.setattr(owner.server, "track", revoke_after_track)
        with pytest.raises(http.client.RemoteDisconnected):
            request(path)
        assert checked.wait(1)
    assert requests == []
