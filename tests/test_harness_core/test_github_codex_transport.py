"""Real bounded subprocess protocol tests; fixture never contacts providers/GitHub."""

import asyncio
import hashlib
import json
import os
import sys
import time

import pytest

from slm_training.harness_core.activity_contract import contract_digest
from slm_training.harness_core.github_connector import (
    ConnectorConfig, DeliveryWaiting, host_connector,
)
from slm_training.harness_core.github_codex_transport import CodexAppServerConfig
from slm_training.harness_core.github_document_delivery import DocumentDelivery


SERVER = '''
import json, os, sys, time, subprocess
from pathlib import Path
root = Path(__file__).parent
settings = json.loads((root / "settings.json").read_text())
for line in sys.stdin:
 request = json.loads(line)
 method = request['method']
 with (root / 'calls.jsonl').open('a') as stream:
  stream.write(json.dumps({'method': method, 'params': request.get('params'),
   'pid': os.getpid(), 'inherited': [k for k in ['CODEX_THREAD_ID','CODEX_SESSION_ID'] if k in os.environ]})+'\\n')
 if 'id' not in request: continue
 if method == 'initialize': result = {}
 elif method == 'thread/start': result = {'thread': {'id': 'owned-probe'}}
 elif method == 'thread/resume':
  assert request['params']['threadId'] == 'owned-probe'
  result = {'thread': {'id': 'owned-probe'}}
 elif method == 'mcpServerStatus/list':
  if settings.get('mode') == 'orphan':
   child = subprocess.Popen([sys.executable, '-c',
    'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(20)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
   (root / 'descendant.pid').write_text(str(child.pid))
   time.sleep(0.1)
  result = {'data': [{'name': 'codex_apps', 'tools': settings['tools']}], 'nextCursor': None}
 elif method == 'mcpServer/tool/call':
  if settings.get('mode') in ['lost', 'orphan']: sys.exit(0)
  if settings.get('mode') == 'hang': time.sleep(20)
  if settings.get('mode') == 'interactive':
   print(json.dumps({'id': 999, 'method': 'mcpServer/elicitation/request', 'params': {}}),flush=True)
   continue
  result = {'isError': settings.get('mode') == 'rejected', 'structuredContent': {'sha': 'a'*40}}
 else: raise RuntimeError('forbidden RPC')
 print(json.dumps({'id': request['id'], 'result': result}), flush=True)
'''


@pytest.fixture
def transport(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_THREAD_ID', 'unknown-parent')
    monkeypatch.setenv('CODEX_SESSION_ID', 'unknown-session')
    executable = tmp_path / 'codex'
    executable.write_text('#!' + sys.executable + '\n' + SERVER)
    executable.chmod(0o700)
    schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
    names = ['github_get_pr_info', 'github_create_commit']
    host = CodexAppServerConfig(
        executable=str(executable), executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        codex_home=str(tmp_path), state_dir=str(tmp_path / 'state'),
        connector_id='connector_76869538009648d5b282a4bb21c3d157', link_id='approved-link')
    tools = {name: 'github.' + name[7:] for name in names}
    config = ConnectorConfig(codex_app_server=host, tool_names=tools,
                             tool_schema_sha256={name: contract_digest(schema) for name in names})
    settings = {'tools': {mapped: {'name': mapped, 'inputSchema': schema,
        '_meta': {'connector_id': host.connector_id, 'link_id': host.link_id}} for mapped in tools.values()}}
    (tmp_path / 'settings.json').write_text(json.dumps(settings))
    return config, tmp_path


def rows(root):
    path = root / 'calls.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def mode(root, value):
    path = root / 'settings.json'
    settings = json.loads(path.read_text())
    settings['mode'] = value
    path.write_text(json.dumps(settings))


def run(config, tool='github_get_pr_info', seconds=12):
    async def invoke():
        async with host_connector(config, allowed_tools={tool}, timeout_seconds=seconds) as call:
            return await call(tool, {})
    return asyncio.run(invoke())


def test_owned_thread_reconnect_and_no_inherited_session(transport):
    config, root = transport
    assert run(config) == run(config)
    calls = rows(root)
    assert len({r['pid'] for r in calls}) == 2
    assert sum(r['method'] == 'thread/start' for r in calls) == 1
    assert sum(r['method'] == 'thread/resume' for r in calls) == 1
    assert not any(r['inherited'] for r in calls)
    assert not any(r['method'] == 'turn/start' for r in calls)
    for pid in {r['pid'] for r in calls}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize('change', ['schema', 'link', 'mapping', 'binary'])
def test_pins_fail_before_dispatch(transport, change):
    config, root = transport
    if change == 'schema':
        config.tool_schema_sha256['github_get_pr_info'] = '0' * 64
    elif change == 'link':
        config.codex_app_server.link_id = 'other-account'
    elif change == 'mapping':
        config.tool_names['github_get_pr_info'] = 'github.create_commit'
    else:
        config.codex_app_server.executable_sha256 = '0' * 64
    with pytest.raises(DeliveryWaiting):
        run(config)
    assert not any(r['method'] == 'mcpServer/tool/call' for r in rows(root))


def test_mutation_loss_stays_pending_and_never_blind_retries(transport):
    config, root = transport
    mode(root, 'lost')
    journal = {}
    saves = []

    async def attempt():
        async with host_connector(config, allowed_tools={'github_create_commit'}, timeout_seconds=12) as call:
            owner = DocumentDelivery({'repository': 'owner/repo'}, journal, connector=call,
                transport=config, save=lambda value: saves.append(json.loads(json.dumps(value))),
                validate=lambda **kwargs: None, verify=None)
            await owner._write('commit', 'github_create_commit', {})

    with pytest.raises(DeliveryWaiting, match='outcome_unknown'):
        asyncio.run(attempt())
    assert saves[-1]['writes']['commit']['state'] == 'pending'
    mode(root, 'normal')
    with pytest.raises(DeliveryWaiting, match='requires_remote_reconciliation'):
        asyncio.run(attempt())
    assert sum(r['method'] == 'mcpServer/tool/call' for r in rows(root)) == 1


@pytest.mark.parametrize('behavior', ['interactive', 'rejected'])
def test_interactive_and_rejected_are_not_success(transport, behavior):
    config, root = transport
    mode(root, behavior)
    if behavior == 'interactive':
        with pytest.raises(DeliveryWaiting, match='interactive'):
            run(config)
    else:
        assert run(config)['isError'] is True


def test_timeout_kills_process_and_retains_owned_thread(transport, monkeypatch):
    from slm_training.harness_core import github_codex_transport
    config, root = transport
    monkeypatch.setattr(github_codex_transport, 'KILL_GRACE_SECONDS', 0.1)
    mode(root, 'hang')
    with pytest.raises(DeliveryWaiting, match='outcome_unknown'):
        run(config, seconds=0.5)
    assert (root / 'state/thread.json').is_file()
    for pid in {r['pid'] for r in rows(root)}:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_mutually_exclusive_and_legacy_identity(transport):
    config, _ = transport
    with pytest.raises(ValueError, match='mutually_exclusive'):
        ConnectorConfig(endpoint='https://host.invalid', codex_app_server=config.codex_app_server)
    with pytest.raises(ValueError, match='mutually_exclusive'):
        ConnectorConfig(authorization_env='TOKEN', codex_app_server=config.codex_app_server)
    assert 'codex_app_server' not in ConnectorConfig().model_dump(mode='json')
    assert json.loads(ConnectorConfig().model_dump_json()) == {
        'endpoint': None, 'authorization_env': None, 'tool_names': {},
        'tool_schema_sha256': {}, 'max_payload_bytes': 200000}


def test_changed_owned_state_never_borrows_thread(transport):
    config, root = transport
    run(config)
    path = root / 'state/thread.json'
    state = json.loads(path.read_text())
    state['binding'] = 'other-host'
    path.write_text(json.dumps(state))
    with pytest.raises(DeliveryWaiting, match='binding_changed'):
        run(config)
    assert not any(r['method'] == 'thread/resume' for r in rows(root))


def test_elapsed_hash_budget_prevents_process_launch(transport, monkeypatch):
    from slm_training.harness_core import github_codex_transport as owner
    config, root = transport
    original = owner._executable

    def slow_hash(host):
        time.sleep(0.2)
        return original(host)

    monkeypatch.setattr(owner, 'KILL_GRACE_SECONDS', 0.1)
    monkeypatch.setattr(owner, '_executable', slow_hash)
    with pytest.raises(DeliveryWaiting, match='deadline_exhausted'):
        run(config, seconds=0.25)
    assert rows(root) == []


def test_process_startup_debits_rpc_and_cleanup_budget(transport, monkeypatch):
    from slm_training.harness_core import github_codex_transport as owner
    config, root = transport
    original = asyncio.create_subprocess_exec

    async def slow_spawn(*args, **kwargs):
        await asyncio.sleep(0.25)
        return await original(*args, **kwargs)

    monkeypatch.setattr(owner, 'KILL_GRACE_SECONDS', 0.1)
    monkeypatch.setattr(asyncio, 'create_subprocess_exec', slow_spawn)
    mode(root, 'hang')
    started = time.monotonic()
    with pytest.raises(DeliveryWaiting, match='outcome_unknown'):
        run(config, seconds=0.5)
    assert time.monotonic() - started < 0.65
    assert any(row['method'] == 'mcpServer/tool/call' for row in rows(root))


def test_exited_root_cleans_observed_detached_descendant(transport, monkeypatch):
    from slm_training.harness_core import github_codex_transport as owner
    from slm_training.harness_core.bounded_process import _process_start_identity
    config, root = transport
    monkeypatch.setattr(owner, 'KILL_GRACE_SECONDS', 0.2)
    mode(root, 'orphan')
    with pytest.raises(DeliveryWaiting, match='outcome_unknown'):
        run(config)
    child = int((root / 'descendant.pid').read_text())
    deadline = time.monotonic() + 1
    while _process_start_identity(child) is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _process_start_identity(child) is None


def test_cleanup_refuses_recycled_identity(monkeypatch):
    from slm_training.harness_core import bounded_process, github_codex_transport
    tree = object.__new__(bounded_process.OwnedProcessTree)
    tree.root = 987654321
    tree.identities = {tree.root: 'original-start', 987654322: 'original-child'}
    monkeypatch.setattr(bounded_process, '_process_start_identity', lambda pid: 'recycled-start')
    signals = []
    monkeypatch.setattr(os, 'kill', lambda *args: signals.append(args))
    monkeypatch.setattr(os, 'killpg', lambda *args: signals.append(args))

    class Reaped:
        pid = tree.root

        async def wait(self):
            return 0

    async def cleanup():
        await github_codex_transport._stop(Reaped(), tree, asyncio.get_running_loop().time() + 1)

    asyncio.run(cleanup())
    assert signals == []
