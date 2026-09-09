    root = tmp_path / "root"
    (root / "loops").mkdir(parents=True)
    (root / "loops" / "owned").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="loop_state_symlink"):
        control.stop_supervisor(root, "owned")
    assert not list(external.iterdir())


def test_prepare_release_has_private_metadata_and_strict_source_identity(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("source documentation\n")
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("private metadata")
    (source / ".claude").mkdir()
    (source / ".claude" / "settings.local.json").write_text("private grant")
    (source / ".codex").mkdir()
    (source / ".codex" / "hooks.json").write_text("{\"hook\": true}")
    release, execution, output = (
        tmp_path / name for name in ("release", "execution", "output")
    )
    result = prepare_release(source, release, execution, output)
    assert runtime_source_identity(execution) == result["source_digest"]
    assert not (execution / ".git").exists()
    assert not (execution / ".claude/settings.local.json").exists()
    assert (execution / ".codex/hooks.json").read_text() == "{\"hook\": true}"
    (output / "report.json").write_text("{}")
    assert runtime_source_identity(execution) == result["source_digest"]
    (execution / "README.md").write_text("not an exempt source mutation")
    with pytest.raises(ValueError, match="execution_source_drift"):
        runtime_source_identity(execution)


def test_release_rejects_alias_into_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="must_be_disjoint"):
        prepare_release(
            source, alias / "release", tmp_path / "execution", tmp_path / "output"
        )

