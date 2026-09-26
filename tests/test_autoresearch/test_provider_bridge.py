"""Real fixture transport/security only; no credentials or live repair inference."""

import http.client
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tests.casefiles import case_values
from pydantic import ValidationError

from slm_training.autoresearch.heal.isolation import IsolationSpec, probe_isolation, run_isolated
from slm_training.harness_core.provider_bridge import ProviderBridge, SANDBOX_AUTHORITY
from slm_training.autoresearch.heal.repair_contracts import ProviderEndpoint


@pytest.fixture
def upstream():
    observed = []
    arrived = threading.Event()
    release = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            observed.append((self.path, dict(self.headers), body))
            arrived.set()
            payload = json.loads(body)
            if payload.get("input") == "hang":
                release.wait(5)
                return
            self.send_response(302 if payload.get("input") == "redirect" else 200)
            self.send_header("Location", "http://127.0.0.1:1/off-target")
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"fixture":true}\n\n')

        def log_message(self, *_args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield ProviderEndpoint(port=server.server_port, model="fixture"), observed, arrived
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(1)


def request(path, *, target="/v1/responses", method="POST", host=SANDBOX_AUTHORITY, body=b'{"model":"fixture","input":"hello","stream":true}', extra=b""):
    with socket.socket(socket.AF_UNIX) as client:
        client.settimeout(2)
        client.connect(str(path))
        client.sendall(f"{method} {target} HTTP/1.1\r\nHost: {host}\r\nContent-Length: {len(body)}\r\n".encode() + extra + b"\r\n" + body)
        response = http.client.HTTPResponse(client)
        response.begin()
        return response.status, response.read(), response.getheaders()


def test_fixed_destination_stream_and_no_worker_credentials(upstream):
    endpoint, observed, _ = upstream
    with ProviderBridge(endpoint, 5, threading.Event()) as path:
        status, body, headers = request(path, extra=b"Authorization: worker-secret\r\nProxy-Authorization: other-secret\r\nX-Forwarded-Host: evil\r\n")
        assert status == 200 and b'"fixture":true' in body
        assert not any(name.lower() == "location" for name, _ in headers)
    assert not path.exists()
    assert len(observed) == 1 and observed[0][0] == "/v1/responses"
    assert observed[0][1]["Host"] == f"127.0.0.1:{endpoint.port}"
    assert not {"Authorization", "Proxy-Authorization", "X-Forwarded-Host"}.intersection(observed[0][1])
    assert observed[0][2] == b'{"model":"fixture","input":"hello","stream":true}'


@pytest.mark.parametrize("options", [
    {"target": "/other"}, {"target": "http://127.0.0.1:1/v1/responses"},
    {"target": "/v1/../responses"}, {"target": "/v1/%72esponses"},
    {"target": "//v1/responses"},
    {"target": "/v1/responses?url=http://127.0.0.1:1"},
    {"method": "CONNECT"}, {"method": "DELETE"}, {"method": "GET"},
    {"host": "evil"}, {"extra": b"Host: evil\r\n"},
    {"extra": b"Upgrade: websocket\r\n"},
])
def test_off_target_requests_never_reach_host_proxy(upstream, options):
    endpoint, observed, _ = upstream
    with ProviderBridge(endpoint, 5, threading.Event()) as path:
        assert request(path, **options)[0] in {403, 501}
    assert observed == []


@pytest.mark.parametrize("extra", [b"Content-Length: 999\r\n", b"Transfer-Encoding: chunked\r\n"])
def test_ambiguous_framing_cannot_smuggle_requests(upstream, extra):
    endpoint, observed, _ = upstream
    with ProviderBridge(endpoint, 5, threading.Event()) as path:
        with pytest.raises(http.client.RemoteDisconnected):
            request(path, extra=extra)
    assert observed == []


def test_redirect_is_rejected_without_following(upstream):
    endpoint, observed, _ = upstream
    with ProviderBridge(endpoint, 5, threading.Event()) as path:
        assert request(path, body=b'{"model":"fixture","input":"redirect"}')[0] == 502
    assert len(observed) == 1


@pytest.mark.parametrize("cancel", [False, True])
def test_cancellation_or_deadline_closes_inflight_connections(upstream, cancel):
    endpoint, _, arrived = upstream
    cancelled = threading.Event()
    bridge = ProviderBridge(endpoint, 10 if cancel else 0.3, cancelled)
    with bridge as path, socket.socket(socket.AF_UNIX) as client:
        client.settimeout(2)
        client.connect(str(path))
        body = b'{"model":"fixture","input":"hang"}'
        client.sendall(f"POST /v1/responses HTTP/1.1\r\nHost: {SANDBOX_AUTHORITY}\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        assert arrived.wait(1)
        if cancel:
            cancelled.set()
        assert client.recv(1024) == b""
    assert not bridge.thread.is_alive() and not bridge.watcher.is_alive()
    assert not path.exists()


@pytest.mark.parametrize("update", case_values(__file__, "test_endpoint_grant_rejects_noncanonical_routes"))
def test_endpoint_grant_rejects_noncanonical_routes(update):
    with pytest.raises(ValidationError):
        ProviderEndpoint(**({"port": 9999, "model": "fixture"} | update))


def test_real_namespace_reaches_only_bridge(upstream, tmp_path):
    capability = probe_isolation()
    assert capability.available, capability.reason
    endpoint, observed, _ = upstream
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    code = (
        "import http.client,socket; "
        f"c=http.client.HTTPConnection('{SANDBOX_AUTHORITY}'); "
        "c.request('POST','/v1/responses',body='{\"model\":\"fixture\"}'); r=c.getresponse(); "
        "assert r.status==200 and b'fixture' in r.read(); "
        f"s=socket.socket(); s.settimeout(.2); assert s.connect_ex(('127.0.0.1',{endpoint.port}))!=0; "
        "s.close(); print('isolated-fixture-ok')"
    )
    with ProviderBridge(endpoint, 10, threading.Event()) as path:
        result = run_isolated(IsolationSpec(workspace, timeout_seconds=8, provider_socket=path),
                              ("/usr/bin/python3", "-I", "-S", "-c", code))
    assert result.returncode == 0, result.stderr
    assert "isolated-fixture-ok" in result.stdout and len(observed) == 1


def test_recovery_config_roundtrip_and_executor_endpoint_wiring(tmp_path, monkeypatch):
    from tests.test_autoresearch.test_repair_dispatch import repair_request as make_request, executor as make_executor
    from slm_training.autoresearch.heal.recovery_dispatch import RecoveryConfig, load_recovery_config

    original = make_request.__wrapped__(tmp_path)
    legacy = original.model_dump_json()
    assert "provider_endpoint" not in legacy
    assert original.model_validate_json(legacy).digest() == original.digest()
    grant = original.grant.model_copy(update={"network": "approved_provider_only", "provider_endpoint": ProviderEndpoint(port=9999, model="fixture-model")})
    config = RecoveryConfig(grant=grant, verifier_release="a" * 64)
    path = tmp_path / "recovery.json"
    path.write_text(config.model_dump_json())
    restored = load_recovery_config(path)
    assert restored.grant == grant and grant.digest() != original.grant.digest()
    executor = make_executor.__wrapped__(monkeypatch)
    observed = []
    run = executor.runner.run
    def capture(request, argv, **kwargs):
        observed.extend(argv)
        return run(request, argv, **kwargs)
    monkeypatch.setattr(executor.runner, "run", capture)
    executor.execute(original.model_copy(update={"grant": restored.grant}), progress=lambda: None, cancelled=lambda: False)
    assert 'model_provider="slm_repair"' in observed
    assert 'model="fixture-model"' in observed
    assert f'model_providers.slm_repair.base_url="http://{SANDBOX_AUTHORITY}/v1"' in observed
    assert 'model_providers.slm_repair.requires_openai_auth=false' in observed
    assert "--ignore-user-config" in observed


def test_endpoint_cannot_be_added_without_network_grant(tmp_path):
    from tests.test_autoresearch.test_repair_dispatch import repair_request as make_request
    from slm_training.autoresearch.heal.repair_contracts import RepairGrant

    original = make_request.__wrapped__(tmp_path).grant.model_dump()
    original["provider_endpoint"] = ProviderEndpoint(port=9999, model="fixture")
    with pytest.raises(ValidationError, match="approved_provider_only"):
        RepairGrant.model_validate(original)


@pytest.mark.parametrize("cancel", [False, True])
def test_actual_runner_grant_mounts_transport_and_tears_it_down(upstream, tmp_path, cancel):
    import hashlib
    from tests.test_autoresearch.test_repair_dispatch import repair_request as make_request
    from slm_training.autoresearch.heal.isolated_agent import BubblewrapAgentRunner
    from slm_training.autoresearch.heal.isolation_workspace import manifest_digest, tree_manifest

    endpoint, observed, arrived = upstream
    source, runtime = tmp_path / "source", tmp_path / "runtime"
    source.mkdir()
    runtime.mkdir()
    (source / "module.py").write_text("answer = 0\n")
    executable = runtime / "agent-fixture"
    executable.write_text(
        "#!/usr/bin/python3\nimport http.client,json,time\nfrom pathlib import Path\n"
        f"c=http.client.HTTPConnection('{SANDBOX_AUTHORITY}')\n"
        "c.request('POST','/v1/responses',body='{\"model\":\"fixture\"}')\nassert c.getresponse().status==200\n"
        + ("time.sleep(30)\n" if cancel else "Path('/workspace/repair-output/proposal.json').write_text(json.dumps({'fixture':True}))\n")
    )
    executable.chmod(0o700)
    request_value = make_request.__wrapped__(tmp_path)
    grant = request_value.grant.model_copy(update={"executable": str(executable), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(), "network": "approved_provider_only", "provider_endpoint": endpoint, "interrupt_seconds": 30})
    request_value = request_value.model_copy(update={"grant": grant, "allowed_paths": ("module.py",), "blocker": request_value.blocker.model_copy(update={"source_digest": manifest_digest(tree_manifest(source))})})
    runner = BubblewrapAgentRunner(source=source, attempt_root=tmp_path / "attempts", runtime_roots=(runtime,))
    assert runner.capability(request_value) is None
    result = runner.run(request_value, (str(executable),), inputs={}, progress=lambda: None,
                        cancelled=lambda: cancel and arrived.is_set())
    assert len(observed) == 1, result
    assert result.outcome == ("cancelled" if cancel else "completed")
    if not cancel:
        assert json.loads(result.final_json) == {"fixture": True}


def _descendants(pid):
    children = Path(f"/proc/{pid}/task/{pid}/children")
    direct = [int(value) for value in children.read_text().split()] if children.exists() else []
    return direct + [descendant for child in direct for descendant in _descendants(child)]


def _alive(pid):
    status = Path(f"/proc/{pid}/stat")
    try:
        return status.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_parent_death_revokes_bridge_and_kills_namespace(upstream, tmp_path):
    endpoint, _, arrived = upstream
    workspace = tmp_path / "private"
    workspace.mkdir()
    child_code = (
        "import http.client,time; "
        f"c=http.client.HTTPConnection('{SANDBOX_AUTHORITY}'); "
        "c.request('POST','/v1/responses',body='{\"model\":\"fixture\"}'); c.getresponse().read(); time.sleep(30)"
    )
    parent_code = (
        "import threading; from pathlib import Path; "
        "from slm_training.harness_core.provider_bridge import ProviderBridge; "
        "from slm_training.autoresearch.heal.repair_contracts import ProviderEndpoint; "
        "from slm_training.autoresearch.heal.isolation import IsolationSpec,run_isolated;\n"
        f"with ProviderBridge(ProviderEndpoint(port={endpoint.port}, model='fixture'), 15, threading.Event()) as path:\n"
        " print(path,flush=True)\n"
        f" run_isolated(IsolationSpec(Path({str(workspace)!r}),provider_socket=path,timeout_seconds=12),"
        f"('/usr/bin/python3','-I','-S','-c',{child_code!r}))\n"
    )
    project = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(project / "src"), str(project)))
    parent = subprocess.Popen(
        [sys.executable, "-B", "-c", parent_code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
    )
    descendants, path = [], None
    try:
        import select
        assert select.select([parent.stdout], [], [], 5)[0], "parent failed to start bridge"
        path = Path(parent.stdout.readline().strip())
        assert arrived.wait(5), "namespace never reached fixture"
        descendants = _descendants(parent.pid)
        assert descendants
        parent.kill()
        parent.wait(timeout=2)
        with socket.socket(socket.AF_UNIX) as client, pytest.raises(ConnectionRefusedError):
            client.connect(str(path))
        deadline = time.monotonic() + 3
        while any(_alive(pid) for pid in descendants) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not any(_alive(pid) for pid in descendants)
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout=2)
        for pid in descendants:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
        # SIGKILL cannot unlink a pathname; it must leave no live socket/process.
        if path is not None and path.name == "provider.sock" and path.parent.name.startswith("slm-provider-"):
            path.unlink(missing_ok=True)
            path.parent.rmdir()


@pytest.mark.parametrize("body", [
    b'{"model":"other","stream":true}', b'{"model":"fixture","background":true}',
    b'{"model":"fixture","background":1}', b'{"model":"fixture","background":"false"}',
    b'{"model":"fixture","model":"other"}', b'{"model":"other","model":"fixture"}',
    b'{"model":"fixture","background":true,"background":false}', b'{}', b'[]',
])
def test_request_body_cannot_expand_model_or_background_grant(upstream, body):
    endpoint, observed, _ = upstream
    with ProviderBridge(endpoint, 5, threading.Event()) as path:
        with pytest.raises(http.client.RemoteDisconnected):
            request(path, body=body)
    assert observed == []
