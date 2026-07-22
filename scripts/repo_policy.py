"""Enforce the repository's tracked-file organization policy."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
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
    ".vercelignore",
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
MAX_PUBLISHED_DATA_BYTES = 50 * 1024 * 1024
# Machine-absolute artifact paths make committed evidence unreproducible from a
# clone (330 dangling references incl. foreign /home/codex/... paths shipped
# before this guard). Applies to NEW design records only; history is immutable.
ABSOLUTE_ARTIFACT_PATH_RE = re.compile(r'"(?:/home/|/Users/|/root/|/tmp/)[^"]*"')
WORKFLOW_TIMEOUT_RE = re.compile(r"^(\s*timeout-minutes:\s*)\d+(\s*)$", re.MULTILINE)
WORKFLOW_JOB_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")


def canonical_run_minutes(*, root: Path = ROOT) -> int:
    """Read the policy constant without importing the not-yet-installed package."""
    path = root / "src/slm_training/levers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "MAX_RUN_MINUTES"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            return node.value.value
    raise ValueError(f"MAX_RUN_MINUTES must be an integer literal in {path}")


def _canonical_string_tuple(name: str, *, root: Path = ROOT) -> tuple[str, ...]:
    """Read one deployment tuple without importing the package."""
    path = root / "src/slm_training/levers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
                return value
    raise ValueError(
        f"{name} must be a string tuple literal in {path}"
    )


def canonical_vercel_include_files(*, root: Path = ROOT) -> tuple[str, ...]:
    """Read the deployment evidence contract from the canonical lever module."""
    return _canonical_string_tuple("VERCEL_FUNCTION_INCLUDE_FILES", root=root)


def canonical_vercel_exclude_files(*, root: Path = ROOT) -> tuple[str, ...]:
    """Read the deployment exclusion contract from the canonical lever module."""
    return _canonical_string_tuple("VERCEL_FUNCTION_EXCLUDE_FILES", root=root)


def _vercel_include_glob(*, root: Path = ROOT) -> str:
    return "{" + ",".join(canonical_vercel_include_files(root=root)) + "}"


def _vercel_exclude_glob(*, root: Path = ROOT) -> str:
    return "{" + ",".join(canonical_vercel_exclude_files(root=root)) + "}"


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


def validate_published_data_sizes(
    paths: Iterable[str], *, root: Path = ROOT
) -> list[str]:
    prefix = "src/slm_training/resources/data/"
    errors = []
    for relative in paths:
        path = root / relative
        if (
            relative.startswith(prefix)
            and path.is_file()
            and path.stat().st_size >= MAX_PUBLISHED_DATA_BYTES
        ):
            errors.append(f"published data file exceeds 50 MiB Git cap: {relative}")
    return errors


def _workflow_jobs(text: str) -> list[tuple[str, int, int]]:
    lines = text.splitlines(keepends=True)
    jobs_line = next((i for i, line in enumerate(lines) if line.rstrip() == "jobs:"), None)
    if jobs_line is None:
        return []
    end = next(
        (i for i in range(jobs_line + 1, len(lines)) if lines[i].strip() and not lines[i][0].isspace()),
        len(lines),
    )
    starts = [
        (match.group(1), i)
        for i in range(jobs_line + 1, end)
        if (match := WORKFLOW_JOB_RE.match(lines[i].rstrip()))
    ]
    return [
        (name, start, starts[index + 1][1] if index + 1 < len(starts) else end)
        for index, (name, start) in enumerate(starts)
    ]


def validate_workflow_timeouts(*, root: Path = ROOT) -> list[str]:
    max_run_minutes = canonical_run_minutes(root=root)
    errors: list[str] = []
    for path in sorted((root / ".github/workflows").glob("*.y*ml")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        relative = path.relative_to(root)
        for job, start, end in _workflow_jobs(text):
            values = [
                int(match.group(1))
                for line in lines[start + 1 : end]
                if (match := re.match(r"^    timeout-minutes:\s*(\d+)\s*$", line))
            ]
            location = f"{relative}#{job}"
            if not values:
                errors.append(f"workflow job lacks canonical run timeout: {location}")
            elif any(value != max_run_minutes for value in values):
                errors.append(
                    f"workflow timeout differs from canonical {max_run_minutes}-minute "
                    f"cap: {location}; run `python -m scripts.repo_policy "
                    "--sync-workflow-timeouts`"
                )
    return errors


def sync_workflow_timeouts(*, root: Path = ROOT) -> list[Path]:
    """Rewrite workflow timeout adapters from the one canonical run-policy lever."""
    max_run_minutes = canonical_run_minutes(root=root)
    changed: list[Path] = []
    for path in sorted((root / ".github/workflows").glob("*.y*ml")):
        before = path.read_text(encoding="utf-8")
        after = WORKFLOW_TIMEOUT_RE.sub(
            rf"\g<1>{max_run_minutes}\g<2>",
            before,
        )
        lines = after.splitlines(keepends=True)
        for _, start, end in reversed(_workflow_jobs(after)):
            if not any(
                re.match(r"^    timeout-minutes:", line)
                for line in lines[start + 1 : end]
            ):
                lines.insert(start + 1, f"    timeout-minutes: {max_run_minutes}\n")
        after = "".join(lines)
        if after != before:
            path.write_text(after, encoding="utf-8")
            changed.append(path)
    return changed


def validate_vercel_run_policy(*, root: Path = ROOT) -> list[str]:
    path = root / "vercel.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("functions", {}).get("src/slm_training/web/vercel.py", {})
    expected_seconds = canonical_run_minutes(root=root) * 60
    expected_files = _vercel_include_glob(root=root)
    expected_excludes = _vercel_exclude_glob(root=root)
    errors: list[str] = []
    if config.get("maxDuration") != expected_seconds:
        errors.append(
            f"Vercel duration differs from canonical {expected_seconds}-second cap; "
            "run `python -m scripts.repo_policy --sync-run-policy`"
        )
    if config.get("includeFiles") != expected_files:
        errors.append(
            "Vercel bundle omits canonical committed evidence; run "
            "`python -m scripts.repo_policy --sync-run-policy`"
        )
    if config.get("excludeFiles") != expected_excludes:
        errors.append(
            "Vercel bundle exclusions differ from canonical levers; run "
            "`python -m scripts.repo_policy --sync-run-policy`"
        )
    return errors


def sync_run_policy(*, root: Path = ROOT) -> list[Path]:
    """Regenerate every external adapter from the canonical run levers."""
    changed = sync_workflow_timeouts(root=root)
    path = root / "vercel.json"
    if not path.is_file():
        return changed
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.setdefault("functions", {}).setdefault(
        "src/slm_training/web/vercel.py", {}
    )
    config["maxDuration"] = canonical_run_minutes(root=root) * 60
    config["includeFiles"] = _vercel_include_glob(root=root)
    config["excludeFiles"] = _vercel_exclude_glob(root=root)
    before = path.read_text(encoding="utf-8")
    after = json.dumps(payload, indent=2) + "\n"
    if after != before:
        path.write_text(after, encoding="utf-8")
        changed.append(path)
    return changed


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


def validate_new_design_record_paths(*, root: Path = ROOT) -> list[str]:
    """Reject machine-absolute paths in newly added docs/design JSON records.

    Scope is git-diff based (added vs HEAD, plus untracked) so committed
    history stays untouched while new evidence must reference repo-relative
    or remote artifact locations.
    """
    added = set(
        _git(
            [
                "diff",
                "--name-only",
                "--diff-filter=A",
                "HEAD",
                "--",
                "docs/design/*.json",
            ],
            root=root,
        ).splitlines()
    )
    added.update(
        _git(
            ["ls-files", "--others", "--exclude-standard", "--", "docs/design/*.json"],
            root=root,
        ).splitlines()
    )
    errors: list[str] = []
    for rel in sorted(added):
        path = root / rel
        if not path.is_file():
            continue
        match = ABSOLUTE_ARTIFACT_PATH_RE.search(
            path.read_text(encoding="utf-8", errors="replace")
        )
        if match:
            errors.append(
                f"machine-absolute artifact path in new design record {rel}: "
                f"{match.group(0)[:80]} (use repo-relative or remote URIs)"
            )
    return errors


def validate_repository(*, root: Path = ROOT) -> list[str]:
    paths = repository_paths(root=root)
    errors = validate_top_level(paths)
    errors.extend(validate_published_data_sizes(paths, root=root))
    errors.extend(validate_workflow_timeouts(root=root))
    errors.extend(validate_vercel_run_policy(root=root))
    ignored = _git(["ls-files", "-ci", "--exclude-standard"], root=root).splitlines()
    errors.extend(
        f"tracked ignored artifact: {path}"
        for path in ignored
        if path not in ALLOWED_TRACKED_IGNORED
    )
    errors.extend(validate_skill_mirrors(root))
    errors.extend(validate_new_design_record_paths(root=root))
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
        "--sync-workflow-timeouts",
        action="store_true",
        help="derive every workflow timeout from slm_training.levers.MAX_RUN_MINUTES",
    )
    parser.add_argument(
        "--sync-run-policy",
        action="store_true",
        help="derive workflow and Vercel adapters from slm_training.levers",
    )
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
    if args.sync_run_policy:
        for path in sync_run_policy():
            print(f"synced {path.relative_to(ROOT)}")
        return 0
    if args.sync_workflow_timeouts:
        for path in sync_workflow_timeouts():
            print(f"synced {path.relative_to(ROOT)}")
        return 0
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
