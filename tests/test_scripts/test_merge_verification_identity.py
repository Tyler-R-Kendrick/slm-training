"""Runtime identity binding avoids rescanning controller-pinned trees."""

def test_controller_digest_skips_javascript_runtime_walk(tmp_path, monkeypatch):
    from scripts import merge_verification_identity as identity

    monkeypatch.setattr(identity.importlib.metadata, "distributions", lambda **_: [])
    modules = tmp_path / "node_modules"
    package = modules / "@agentv" / "core"
    package.mkdir(parents=True)
    lock = tmp_path / "package-lock.json"
    lock.write_text('{"lockfileVersion": 3}\n')
    monkeypatch.setenv("AGENTV_NODE_MODULES", str(modules))

    def fail_if_rescanned(*_):
        raise AssertionError("controller runtime was scanned twice")

    monkeypatch.setattr(identity, "runtime_identity", fail_if_rescanned)
    result = identity.environment_identity(runtime_identity_value="a" * 64)

    runtime = result["javascript_runtime_dependencies"]
    assert runtime["trees"] == {"controller_runtime_identity": "a" * 64}
    assert runtime["package_locks"][str(lock)] == identity.file_digest(lock)
