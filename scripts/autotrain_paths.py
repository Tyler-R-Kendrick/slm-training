"""Filesystem layout of the continuous autotrain loop.

Every path the loop reads or writes, and the predicates that classify a path as
loop-owned, foreign-dirty, or part of a closeout. One responsibility: knowing
where things live, so no caller rebuilds a path by string concatenation.

Extracted from ``scripts/run_autotrain_continuous.py``.
See ``docs/design/code-quality-contract.md``.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

PROMOTE_EXPECTATIONS_REL = Path(
    "src/slm_training/resources/experiments/autotrain_climb/"
    "metric_expectations.promote.v1.json"
)

SCREENING_EXPECTATIONS_REL = Path(
    "src/slm_training/resources/experiments/autotrain_climb/"
    "metric_expectations.screening.v1.json"
)

DRIVER_LOCK_BASENAME = "driver.lock"


def champion_queue_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "champion_queue.jsonl"


def driver_lock_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / DRIVER_LOCK_BASENAME


def terminal_verdict_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "terminal_verdict.json"


def merge_head_path(cwd: Path) -> Path | None:
    """Return MERGE_HEAD path when a merge is in progress, else None."""
    try:
        rel = subprocess.check_output(
            ["git", "rev-parse", "--git-path", "MERGE_HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if not rel:
        return None
    path = Path(rel)
    if not path.is_absolute():
        path = cwd / path
    return path if path.is_file() else None


def normalize_repo_relpath(rel: str) -> str:
    """POSIX repo-relative path: strip a ``./`` prefix, keep leading-dot files."""
    # ponytail: str.lstrip("./" ) is a charset strip and would turn ".serena" into "serena".
    return rel.replace("\\", "/").lstrip().removeprefix("./")


def is_continuous_closeout_path(rel: str) -> bool:
    """True when a dirty path is continuous-driver closeout material only."""
    path = normalize_repo_relpath(rel)
    if path in {"docs/MODEL_CARD.md", "README.md"}:
        return True
    # Family closures are append-only machine-written science (WP-4): the
    # supervisor's conclusion writer appends them and the driver commits them
    # exactly like continuous results docs.
    if path == (
        "src/slm_training/resources/experiments/autotrain_climb/"
        "closed_approaches.v1.json"
    ):
        return True
    if path == "src/slm_training/resources/test_seeds.jsonl":
        return True
    if path.startswith(
        "src/slm_training/resources/data/eval/e938_role_safe_all_targets_smoke6_v1/"
    ):
        return True
    if not path.startswith("docs/design/"):
        return False
    name = path[len("docs/design/") :]
    if "/" in name:
        return False
    return name.startswith("continuous-") and name.endswith((".md", ".json"))


LOOP_OWNED_GENERATED_PATHS = frozenset(
    {
        "src/slm_training/resources/evidence_store/local_index.jsonl",
    }
)

LOOP_OWNED_GENERATED_SUFFIXES = (
    "/evidence_store/local_index.jsonl",
    "/screening_sample_size.json",
)


def is_loop_owned_generated_path(rel: str) -> bool:
    """True when the continuous driver is allowed to dirty this tracked path."""
    path = normalize_repo_relpath(rel)
    if path in LOOP_OWNED_GENERATED_PATHS:
        return True
    return any(path.endswith(suffix) for suffix in LOOP_OWNED_GENERATED_SUFFIXES)


def is_foreign_dirty_path(rel: str) -> bool:
    """True when porcelain path should hard-block continuous thrash."""
    path = normalize_repo_relpath(rel)
    if is_continuous_closeout_path(path):
        return False
    if is_loop_owned_generated_path(path):
        return False
    # Runtime artifacts are never continuous blockers (usually gitignored).
    if path == "outputs" or path.startswith("outputs/"):
        return False
    if path.startswith(".pytest_cache/") or path == ".pytest_cache":
        return False
    # Serena cache/memories are local agent state; tracked config still blocks
    # unless the project.yml rewrite is a comment/whitespace-only strip.
    if path == ".serena" or path.startswith(".serena/"):
        return path in {".serena/project.yml", ".serena/.gitignore"}
    return True


def porcelain_paths(porcelain: str) -> list[str]:
    """Parse ``git status --porcelain`` paths robustly (XY + space + path)."""
    paths: list[str] = []
    for line in porcelain.splitlines():
        if not line.strip():
            continue
        # Standard format: two-char XY, space, path (optional rename "a -> b").
        if len(line) >= 4 and line[2] == " ":
            body = line[3:]
        else:
            # Fallback: first space-separated field after status token.
            parts = line.split(" ", 1)
            body = parts[1] if len(parts) > 1 else line
        if " -> " in body:
            body = body.split(" -> ", 1)[1]
        path = body.strip().strip('"')
        if path:
            paths.append(path)
    return paths


def continuous_docs_paths(cwd: Path, campaign_id: str) -> tuple[Path, Path]:
    design = cwd / "docs" / "design"
    stem = f"{campaign_id}-results"
    return design / f"{stem}.md", design / f"{stem}.json"


def hillclimb_review_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "hillclimb_stagnation_review.json"


def heal_retired_versions_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "heal_retired_versions.jsonl"


def loop_state_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "state.json"


def loop_champion_dir(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id


def loop_campaign_dirs(root: Path, loop_id: str) -> list[Path]:
    """This loop's campaign dirs, newest cycle first (``_campaign_id`` layout)."""

    digest = hashlib.sha256(loop_id.encode("utf-8")).hexdigest()[:8]
    found: list[tuple[int, str, Path]] = []
    try:
        candidates = list(root.glob(f"continuous-loop-*-{digest}-c*"))
    except OSError:
        return []
    for path in candidates:
        if not path.is_dir():
            continue
        match = re.search(r"-c(\d+)$", path.name)
        if match is None:
            continue
        found.append((int(match.group(1)), path.name, path))
    found.sort(reverse=True)
    return [item[2] for item in found]


def promote_expectations_path() -> Path:
    """Repo-relative locked promote expectations (absolute when repo root known)."""
    # Prefer package resource next to climb policy.
    pkg = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "slm_training"
        / "resources"
        / "experiments"
        / "autotrain_climb"
        / "metric_expectations.promote.v1.json"
    )
    if pkg.is_file():
        return pkg
    cwd_pkg = Path.cwd() / PROMOTE_EXPECTATIONS_REL
    return cwd_pkg


def screening_expectations_path() -> Path:
    """Repo-relative exact-zero expectations for decision-bearing screening."""
    return promote_expectations_path().with_name(SCREENING_EXPECTATIONS_REL.name)


def promotion_replicate_ledger_path(root: Path, loop_id: str) -> Path:
    return root / "loops" / loop_id / "promotion_replicates.jsonl"


def continuous_evidence_roots(
    root: Path, loop_id: str, predecessor_campaign_id: str | None
) -> tuple[Path, ...]:
    roots = [root / "loops" / loop_id, root / "sdlc_delivery_ledger.jsonl"]
    if predecessor_campaign_id:
        roots.insert(0, root / predecessor_campaign_id)
    return tuple(roots)
