from __future__ import annotations

import subprocess
from pathlib import Path

from scripts import update_openwiki


def test_run_openwiki_uses_temporary_link_and_restores_scaffolding(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "docs/openwiki").mkdir(parents=True)
    workflow = tmp_path / ".github/workflows/openwiki-update.yml"
    workflow.parent.mkdir(parents=True)
    protected = [tmp_path / "AGENTS.md", tmp_path / "CLAUDE.md", workflow]
    for path in protected:
        path.write_text("original\n", encoding="utf-8")

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
        assert command == ["openwiki", "code", "--update", "--print"]
        assert kwargs["cwd"] == tmp_path
        assert (tmp_path / "openwiki").resolve() == (tmp_path / "docs/openwiki").resolve()
        for path in protected:
            path.write_text("overwritten\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(update_openwiki.subprocess, "run", fake_run)

    assert update_openwiki.run_openwiki(["--update", "--print"], root=tmp_path) == 0
    assert not (tmp_path / "openwiki").exists()
    assert all(path.read_text(encoding="utf-8") == "original\n" for path in protected)
