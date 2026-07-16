from pathlib import Path

from scripts.repo_policy import (
    MAX_PUBLISHED_DATA_BYTES,
    pre_tool_decision,
    raw_mv_paths,
    validate_skill_mirrors,
    validate_top_level,
    validate_published_data_sizes,
)


def test_published_data_size_cap(tmp_path) -> None:
    relative = "src/slm_training/resources/data/train/huge/records.jsonl"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(b"x")
    assert validate_published_data_sizes([relative], root=tmp_path) == []
    with path.open("r+b") as handle:
        handle.truncate(MAX_PUBLISHED_DATA_BYTES)
    assert validate_published_data_sizes([relative], root=tmp_path) == [
        f"published data file exceeds 50 MiB Git cap: {relative}"
    ]


def test_top_level_allowlist_rejects_sprawl() -> None:
    assert validate_top_level(["src/slm_training/app.py", "scratch/note.py"]) == [
        "unapproved top-level path: scratch"
    ]


def test_former_root_buckets_cannot_return() -> None:
    assert validate_top_level(
        [
            "api/index.py",
            "fixtures/data.json",
            "grammars/openui.lark",
            "openwiki/quickstart.md",
            "tools/dashboard/package.json",
        ]
    ) == [
        "unapproved top-level path: api",
        "unapproved top-level path: fixtures",
        "unapproved top-level path: grammars",
        "unapproved top-level path: openwiki",
        "unapproved top-level path: tools",
    ]


def test_raw_mv_detection_distinguishes_git_mv() -> None:
    assert raw_mv_paths("mv src/old.py src/new.py") == [
        "src/old.py",
        "src/new.py",
    ]
    assert raw_mv_paths("cd src && mv old.py new.py") == [
        "src/old.py",
        "src/new.py",
    ]
    assert raw_mv_paths("git mv src/old.py src/new.py") == []


def test_pre_tool_hook_blocks_only_tracked_moves() -> None:
    payload = {"tool_input": {"command": "mv src/old.py src/new.py"}}
    assert pre_tool_decision(payload, tracked=lambda path: path == "src/old.py") == {
        "decision": "block",
        "reason": "Tracked repository paths must be moved with git mv, not mv.",
    }
    assert pre_tool_decision(payload, tracked=lambda _path: False) is None

    nested = {
        "tool_input": {
            "command": "mv old.py new.py",
            "workdir": "src/slm_training",
        }
    }
    assert pre_tool_decision(
        nested,
        tracked=lambda path: path == "src/slm_training/old.py",
    ) is not None


def test_skill_discovery_entries_must_be_canonical_symlinks(tmp_path: Path) -> None:
    source = tmp_path / ".agents/skills/example"
    source.mkdir(parents=True)
    discovery = tmp_path / ".claude/skills/example"
    discovery.mkdir(parents=True)
    assert validate_skill_mirrors(tmp_path) == [
        "copied skill mirror: .claude/skills/example; use a symlink to ../../.agents/skills/example"
    ]

    discovery.rmdir()
    discovery.symlink_to("../../.agents/skills/example", target_is_directory=True)
    assert validate_skill_mirrors(tmp_path) == []
