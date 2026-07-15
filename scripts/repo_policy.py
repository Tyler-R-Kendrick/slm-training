"""Enforce the repository's tracked-file organization policy."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_ROOTS = {
    ".agents",
    ".claude",
    ".codex",
    ".cursor",
    ".env.example",
    ".githooks",
    ".github",
    ".gitignore",
    ".mcp.json",
    ".nvmrc",
    ".python-version",
    ".rtk",
    ".serena",
    ".vscode",
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    "LICENSE",
    "README.md",
    "RTK.md",
    "VERCEL.md",
    "docs",
    "package-lock.json",
    "package.json",
    "playwright.config.ts",
    "pyproject.toml",
    "scripts",
    "skills-lock.json",
    "src",
    "tests",
    "vercel.json",
}
DISCOVERY_ROOTS = (Path(".claude/skills"), Path(".cursor/skills"))
SHELL_OPERATORS = {"&", "&&", ";", "|", "||"}
ALLOWED_TRACKED_IGNORED = {
    ".agents/skills/huggingface-community-evals/examples/.env.example",
    ".env.example",
}


def validate_top_level(paths: Iterable[str]) -> list[str]:
    roots = {path.split("/", 1)[0] for path in paths if path}
    return [
        f"unapproved top-level path: {name}"
        for name in sorted(roots - ALLOWED_ROOTS)
    ]


def validate_skill_mirrors(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    canonical = root / ".agents/skills"
    for relative_root in DISCOVERY_ROOTS:
        discovery = root / relative_root
        if not discovery.is_dir():
            continue
        for entry in sorted(discovery.iterdir()):
            source = canonical / entry.name
            if not source.is_dir():
                errors.append(f"orphan skill discovery entry: {entry.relative_to(root)}")
                continue
            expected = Path("../../.agents/skills") / entry.name
            if not entry.is_symlink():
                errors.append(
                    f"copied skill mirror: {entry.relative_to(root)}; use a symlink to {expected}"
                )
            elif Path(os.readlink(entry)) != expected:
                errors.append(
                    f"wrong skill symlink: {entry.relative_to(root)} -> {os.readlink(entry)}"
                )

    codex_skills = root / ".codex/skills"
    if codex_skills.exists():
        for entry in sorted(codex_skills.iterdir()):
            errors.append(
                f"redundant Codex skill copy: {entry.relative_to(root)}; Codex loads .agents/skills"
            )
    return errors


def _git(args: list[str], *, root: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def repository_paths(*, root: Path = ROOT) -> list[str]:
    tracked = _git(["ls-files"], root=root).splitlines()
    untracked = _git(
        ["ls-files", "--others", "--exclude-standard"], root=root
    ).splitlines()
    return sorted({*tracked, *untracked})


def validate_repository(*, root: Path = ROOT) -> list[str]:
    errors = validate_top_level(repository_paths(root=root))
    ignored = _git(["ls-files", "-ci", "--exclude-standard"], root=root).splitlines()
    errors.extend(
        f"tracked ignored artifact: {path}"
        for path in ignored
        if path not in ALLOWED_TRACKED_IGNORED
    )
    errors.extend(validate_skill_mirrors(root))
    return errors


def raw_mv_paths(command: str) -> list[str] | None:
    """Return raw-mv operands, [] when absent, or None when parsing is unsafe."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return None if "mv" in command else []

    operands: list[str] = []
    segment: list[str] = []
    command_cwd = Path()
    for token in [*tokens, ";"]:
        if token not in SHELL_OPERATORS:
            segment.append(token)
            continue
        if segment:
            command_index = next(
                (
                    i
                    for i, part in enumerate(segment)
                    if "=" not in part or part.startswith(("/", "./"))
                ),
                0,
            )
            executable = Path(segment[command_index]).name
            if executable == "command" and command_index + 1 < len(segment):
                command_index += 1
                executable = Path(segment[command_index]).name
            if executable == "cd" and command_index + 1 < len(segment):
                destination = Path(segment[command_index + 1])
                command_cwd = (
                    destination
                    if destination.is_absolute()
                    else command_cwd / destination
                )
            if executable == "mv":
                args = segment[command_index + 1 :]
                positional = [arg for arg in args if arg == "-" or not arg.startswith("-")]
                if len(positional) < 2:
                    return None
                operands.extend(
                    str(command_cwd / path) if not Path(path).is_absolute() else path
                    for path in positional
                )
        segment = []
    return operands


def is_tracked_path(path: str, *, root: Path = ROOT) -> bool:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root.resolve())
        except ValueError:
            return False
    result = subprocess.run(
        ["git", "ls-files", "--", candidate.as_posix()],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def hook_command(payload: dict) -> str:
    for value in (
        payload.get("tool_input"),
        payload.get("toolInput"),
        payload.get("tool_args"),
        payload.get("toolArgs"),
    ):
        if isinstance(value, dict) and isinstance(value.get("command"), str):
            return value["command"]
    command = payload.get("command")
    return command if isinstance(command, str) else ""


def hook_workdir(payload: dict) -> Path:
    for value in (payload.get("tool_input"), payload.get("toolInput")):
        if isinstance(value, dict):
            workdir = value.get("workdir") or value.get("cwd")
            if isinstance(workdir, str):
                return Path(workdir)
    workdir = payload.get("cwd")
    return Path(workdir) if isinstance(workdir, str) else Path()


def pre_tool_decision(
    payload: dict,
    *,
    tracked: Callable[[str], bool] = is_tracked_path,
) -> dict | None:
    paths = raw_mv_paths(hook_command(payload))
    if paths is None:
        return {
            "decision": "block",
            "reason": "Could not safely inspect raw mv command. Use git mv for tracked repository paths.",
        }
    workdir = hook_workdir(payload)
    resolved = [str(workdir / path) if not Path(path).is_absolute() else path for path in paths]
    if resolved and any(tracked(path) for path in resolved):
        return {
            "decision": "block",
            "reason": "Tracked repository paths must be moved with git mv, not mv.",
        }
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staged",
        action="store_true",
        help="document that validation is running from the staged pre-commit hook",
    )
    parser.add_argument(
        "--pre-tool-use",
        action="store_true",
        help="read hook JSON from stdin and block raw moves of tracked paths",
    )
    args = parser.parse_args(argv)
    if args.pre_tool_use:
        try:
            payload = json.load(sys.stdin)
        except json.JSONDecodeError as exc:
            print(json.dumps({"decision": "block", "reason": f"Invalid hook input: {exc}"}))
            return 0
        decision = pre_tool_decision(payload)
        if decision:
            print(json.dumps(decision))
        return 0

    errors = validate_repository()
    if errors:
        print("repo-policy: failed", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    scope = "staged" if args.staged else "tracked + untracked"
    print(f"repo-policy: ok ({scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
