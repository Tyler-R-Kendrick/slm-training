"""Read-only observability facade over the repo's on-disk evidence.

Every dashboard view flows through :class:`Readers`. The methods are pure reads
with tolerant JSON parsing and **cold-start fallback**: because ``outputs/`` is
gitignored and empty on a fresh checkout (and on a read-only Vercel deploy), each
reader falls back to the committed source of truth — ``docs/design/*.json``,
``docs/MODEL_CARD.md``, and ``fixtures/``. Every payload carries a ``provenance``
of ``"live"`` (read from ``outputs/``) or ``"committed"`` (fallback snapshot) so
the UI never passes a committed snapshot off as current state — matching the
repo's honesty ethos.

This module must stay import-light (no torch, no model loading) so the Vercel
entrypoint keeps cold-starting.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)
from slm_training.lineage.store import LineageStore
from slm_training.web.comparisons import BlindedComparisonStore
from slm_training.web.deployments import DeploymentRegistry

# docs/design scoreboard file names by matrix kind.
SCOREBOARD_FILES: dict[str, str] = {
    "quality": "quality-matrix-results.json",
    "grammar": "grammar-matrix-results.json",
    "perf": "perf-matrix-results.json",
    "phase": "phase-abc-results.json",
}

# Suite ordering used across the UI (matches ALLOWED_SPLITS eval order).
SUITE_ORDER = ("smoke", "held_out", "adversarial", "ood", "rico_held")
TRACKS = ("twotower", "causal_lm")


def _read_json(path: Path) -> Any | None:
    """Tolerant JSON read: missing/invalid file returns None rather than raising."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_jsonl(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
                if limit is not None and len(rows) >= limit:
                    break
    except OSError:
        return rows
    return rows


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


# Remote monitoring handles come from a dispatch job's captured stdout (the
# `hf jobs run` / SSH output flows into the job log) or a synced run's bucket.
_REMOTE_URL_RE = re.compile(
    r"https://(?:huggingface\.co|[A-Za-z0-9-]+\.hf\.space|[A-Za-z0-9.-]*trackio[A-Za-z0-9.-]*)\S*"
)


def _first_remote_url(text: str | None) -> str | None:
    match = _REMOTE_URL_RE.search(text or "")
    return match.group(0).rstrip(".,)'\"") if match else None


def _read_text_tail(path: Path, nbytes: int = 4000) -> str:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - nbytes))
            return handle.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _results_of(payload: Any) -> list[dict[str, Any]]:
    """Accept either a bare array or a ``{..., "results": [...]}`` wrapper."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list):
            return [r for r in results if isinstance(r, dict)]
    return []


def _parse_markdown_table(text: str, heading: str) -> list[dict[str, str]]:
    """Parse the first GitHub-flavored table appearing under ``heading``.

    Tolerant: returns [] if the heading or a table is not found.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(heading.strip().lower()):
            start = i
            break
    if start is None:
        return []
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("#") and header is not None:
            break  # next section
        if not stripped.startswith("|"):
            if header is not None and rows:
                break
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if set("".join(cells)) <= set("-: "):
            continue  # separator row
        if header is None:
            header = [c.strip().lower() for c in cells]
            continue
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        rows.append(dict(zip(header, cells)))
    return rows


class Readers:
    """Facade over the repo's evidence tree. ``root`` is the repo root."""

    def __init__(self, root: Path | str = Path(".")) -> None:
        self.root = Path(root)
        self.docs_design = self.root / "docs" / "design"
        self.model_card = self.root / "docs" / "MODEL_CARD.md"
        self.outputs = self.root / "outputs"
        self.fixtures = self.root / "fixtures"
        self.lineage = LineageStore(self.outputs / "lineage")
        self.deployments = DeploymentRegistry(self.outputs / "lineage" / "deployments")
        self.comparisons = BlindedComparisonStore(
            self.outputs / "annotations" / "comparisons.jsonl"
        )

    # ---- scoreboards (experiment / perf matrices) -----------------------------

    def scoreboard(self, kind: str) -> dict[str, Any]:
        filename = SCOREBOARD_FILES.get(kind)
        if filename is None:
            return {"kind": kind, "provenance": "unknown", "results": [], "meta": {}}
        payload = _read_json(self.docs_design / filename)
        results = _results_of(payload)
        meta = {k: v for k, v in (payload or {}).items() if k != "results"} if isinstance(
            payload, dict
        ) else {}
        passed = sum(1 for r in results if r.get("pass") is True)
        return {
            "kind": kind,
            "provenance": "committed",
            "reference": f"docs/design/{filename}",
            "meta": meta,
            "count": len(results),
            "passed": passed,
            "results": results,
        }

    def scoreboards(self) -> list[dict[str, Any]]:
        index: list[dict[str, Any]] = []
        for kind in SCOREBOARD_FILES:
            board = self.scoreboard(kind)
            index.append(
                {
                    "kind": kind,
                    "count": board["count"],
                    "passed": board["passed"],
                    "provenance": board["provenance"],
                    "reference": board.get("reference"),
                }
            )
        return index

    # ---- runs (lineage + per-run artifacts) -----------------------------------

    def runs(self) -> dict[str, Any]:
        runs_dir = self.lineage.root / "runs"
        live: list[dict[str, Any]] = []
        if runs_dir.exists():
            for current in sorted(runs_dir.glob("*/current.json")):
                pointer = _read_json(current) or {}
                record = pointer.get("record", "manifest.json")
                manifest = _read_json(current.parent / record) or {}
                if not manifest:
                    continue
                live.append(
                    {
                        "run_id": manifest.get("run_id", current.parent.name),
                        "track": manifest.get("track"),
                        "lifecycle_state": manifest.get("lifecycle_state"),
                        "created_at": manifest.get("created_at"),
                        "metrics": manifest.get("metrics", {}),
                        "artifact_uris": manifest.get("artifact_uris", []),
                    }
                )
        if live:
            return {"provenance": "live", "runs": live}
        # Cold start: synthesize runs from committed matrix rows.
        derived: list[dict[str, Any]] = []
        for kind in ("quality", "grammar", "perf"):
            for row in self.scoreboard(kind)["results"]:
                derived.append(
                    {
                        "run_id": row.get("run_id") or row.get("id"),
                        "experiment_id": row.get("id"),
                        "matrix": kind,
                        "pass": row.get("pass"),
                        "description": row.get("description"),
                        "checkpoint": row.get("checkpoint"),
                        "lifecycle_state": None,
                    }
                )
        return {"provenance": "committed", "runs": derived}

    def _scoreboard_row(self, run_id: str) -> dict[str, Any] | None:
        """Find the matrix row (with suites) whose run_id or experiment id matches."""
        for kind in ("quality", "grammar", "perf"):
            for row in self.scoreboard(kind)["results"]:
                if run_id in (row.get("run_id"), row.get("id")):
                    return {"matrix": kind, **row}
        return None

    def run(self, run_id: str) -> dict[str, Any]:
        run_dir = self.outputs / "runs" / run_id
        live = {
            "train_summary": _read_json(run_dir / "train_summary.json"),
            "gates": _read_json(run_dir / "gates.json"),
            "telemetry": _read_json(run_dir / "train_telemetry.json"),
            "matrix_result": _read_json(run_dir / "matrix_result.json"),
        }
        lineage_manifest = None
        try:
            lineage_manifest = self.lineage.load_run(run_id).to_dict()
        except (FileNotFoundError, OSError, ValueError):
            lineage_manifest = None
        has_live = any(v is not None for v in live.values()) or bool(lineage_manifest)
        # Attach the matching committed scoreboard row so the detail view is useful
        # even on a cold outputs/; derive a gate matrix from its suites when no live
        # gates.json exists.
        scoreboard = self._scoreboard_row(run_id)
        artifacts = dict(live)
        if artifacts["gates"] is None and scoreboard and scoreboard.get("suites"):
            artifacts["gates"] = evaluate_ship_gates(scoreboard["suites"])
        return {
            "run_id": run_id,
            "provenance": "live" if has_live else "committed",
            "manifest": lineage_manifest,
            "scoreboard": scoreboard,
            **artifacts,
        }

    # ---- lineage champions / deployments --------------------------------------

    def champions(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for track in TRACKS:
            try:
                pointer = self.lineage.champion(track)
            except (OSError, ValueError):
                pointer = None
            out[track] = pointer.to_dict() if pointer else None
        return {"provenance": "live", "champions": out}

    def deployment_state(self) -> dict[str, Any]:
        try:
            payload = self.deployments.payload()
        except (OSError, ValueError, KeyError):
            payload = {"selected": None, "tracks": {}}
        return {"provenance": "live", **payload}

    # ---- checkpoints roster ---------------------------------------------------

    def checkpoints(self) -> dict[str, Any]:
        roster: list[dict[str, Any]] = []
        card_text = ""
        try:
            card_text = self.model_card.read_text(encoding="utf-8")
        except OSError:
            card_text = ""
        for row in _parse_markdown_table(card_text, "## Current checkpoint roster"):
            roster.append(
                {
                    "role": row.get("role", ""),
                    "run_id": row.get("run id", "").strip("`"),
                    "kind": row.get("kind", ""),
                    "location": row.get("location", ""),
                    "status": row.get("status", ""),
                    "source": "model_card",
                    "provenance": "committed",
                }
            )
        # Merge live champions / deployment selection.
        champions = self.champions()["champions"]
        for track, pointer in champions.items():
            if pointer:
                roster.append(
                    {
                        "role": f"Champion ({track})",
                        "run_id": pointer.get("run_id", ""),
                        "kind": "lineage champion",
                        "location": pointer.get("artifact_uri", ""),
                        "status": "champion",
                        "source": "lineage",
                        "provenance": "live",
                    }
                )
        # Fixture checkpoints on disk.
        fx = self.fixtures / "checkpoints"
        if fx.exists():
            for ckpt in sorted(fx.glob("*/last.pt")):
                roster.append(
                    {
                        "role": "Fixture checkpoint",
                        "run_id": ckpt.parent.name,
                        "kind": "fixture",
                        "location": str(ckpt.relative_to(self.root)),
                        "status": "demo only",
                        "source": "fixtures",
                        "provenance": "committed",
                    }
                )
        return {"checkpoints": roster, "deployment": self.deployment_state()}

    def gates_for_run(self, run_id: str) -> dict[str, Any]:
        gates = _read_json(self.outputs / "runs" / run_id / "gates.json")
        if gates is not None:
            return {"provenance": "live", **gates}
        # Cold start: find the run's suites in the committed scoreboards and
        # evaluate the default policy so the UI always has a gate matrix.
        for kind in ("quality", "grammar"):
            for row in self.scoreboard(kind)["results"]:
                if run_id in (row.get("run_id"), row.get("id")):
                    suites = row.get("suites") or {}
                    payload = evaluate_ship_gates(suites)
                    return {"provenance": "committed", **payload}
        return {
            "provenance": "committed",
            "policy": DEFAULT_SHIP_GATES,
            "actual": {},
            "gates": {},
            "failures": ["no_scoreboard_for_run"],
            "pass": False,
        }

    # ---- training / test data -------------------------------------------------

    def train_data(self, version: str | None = None) -> dict[str, Any]:
        train_root = self.outputs / "train_data"
        versions = (
            sorted(p.name for p in train_root.iterdir() if p.is_dir())
            if train_root.exists()
            else []
        )
        if not versions:
            return {"provenance": "committed", "versions": [], **self._fixture_data()}
        chosen = version if version in versions else versions[-1]
        vdir = train_root / chosen
        stats = _read_json(vdir / "stats.json") or {}
        manifest = _read_json(vdir / "manifest.json") or {}
        return {
            "provenance": "live",
            "versions": versions,
            "version": chosen,
            "stats": stats,
            "source_families": manifest.get("source_family_stats")
            or manifest.get("source_families")
            or {},
            "record_count": stats.get("record_count"),
        }

    def train_records(
        self, version: str, *, split: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        path = self.outputs / "train_data" / version / "records.jsonl"
        rows = _read_jsonl(path, limit=None)
        if split:
            rows = [r for r in rows if r.get("split") == split]
        return {"version": version, "count": len(rows), "records": rows[:limit]}

    def _fixture_data(self) -> dict[str, Any]:
        """Cold-start corpus health from committed fixtures."""
        sources = {
            "train_seeds": self.fixtures / "train_seeds.jsonl",
            "rico_train": self.fixtures / "rico" / "semantic_train.jsonl",
            "awwwards": self.fixtures / "awwwards" / "sites.jsonl",
            "test_seeds": self.fixtures / "test_seeds.jsonl",
        }
        counts = {name: _count_lines(path) for name, path in sources.items()}
        return {"fixture_counts": counts, "record_count": counts.get("train_seeds")}

    def test_data(self) -> dict[str, Any]:
        test_root = self.outputs / "test_data"
        if test_root.exists():
            versions = sorted(p.name for p in test_root.iterdir() if p.is_dir())
            if versions:
                vdir = test_root / versions[-1]
                suites_dir = vdir / "suites"
                sizes: dict[str, int] = {}
                if suites_dir.exists():
                    for suite in SUITE_ORDER:
                        sizes[suite] = _count_lines(suites_dir / suite / "records.jsonl")
                return {"provenance": "live", "version": versions[-1], "suites": sizes}
        # Cold start: committed fixture test seeds by split.
        rows = _read_jsonl(self.fixtures / "test_seeds.jsonl")
        sizes = {suite: 0 for suite in SUITE_ORDER}
        for r in rows:
            split = r.get("split")
            if split in sizes:
                sizes[split] += 1
        return {"provenance": "committed", "version": None, "suites": sizes}

    # ---- annotations / comparisons -------------------------------------------

    def annotations_summary(self) -> dict[str, Any]:
        ann = self.outputs / "annotations"
        prefs = self.outputs / "preferences"
        return {
            "feedback": _count_lines(ann / "feedback.jsonl"),
            "bad_outputs": _count_lines(ann / "bad_outputs.jsonl"),
            "comparisons": _count_lines(ann / "comparisons.jsonl"),
            "human_pairs": _count_lines(prefs / "human_pairs.jsonl"),
            "human_train": _count_lines(
                self.fixtures / "annotations" / "human_train.jsonl"
            ),
        }

    def comparison_metrics(self, candidate_run_id: str) -> dict[str, Any]:
        from slm_training.lineage.promotion import wilson_lower_bound

        metrics = self.comparisons.metrics(candidate_run_id)
        total = metrics.get("total", 0)
        wins = metrics.get("candidate_wins", 0)
        wilson = wilson_lower_bound(wins, total) if total else 0.0
        win_rate = (wins / total) if total else 0.0
        # Deployment gate policy (mirrors lineage.promotion.deployment_failures).
        checks = {
            "min_comparisons": total >= 100,
            "win_rate_gt_55": win_rate > 0.55,
            "wilson_gt_50": wilson > 0.50,
        }
        return {
            "candidate_run_id": candidate_run_id,
            **metrics,
            "win_rate": win_rate,
            "wilson_lower_bound": wilson,
            "deployment_ready": all(checks.values()),
            "checks": checks,
        }

    # ---- remote dispatch monitoring ------------------------------------------

    def dispatches(self) -> dict[str, Any]:
        """Surface dispatched (remote GPU) trains: dispatch-kind jobs + their
        parsed remote handle, plus durable remotes from synced runs. Reads the
        persisted job meta/log so it works read-only and after a restart."""
        jobs: list[dict[str, Any]] = []
        jobs_dir = self.outputs / "jobs"
        if jobs_dir.exists():
            for meta_path in sorted(jobs_dir.glob("*/meta.json"), reverse=True):
                meta = _read_json(meta_path) or {}
                if meta.get("kind") != "dispatch":
                    continue
                remote_url = _first_remote_url(
                    _read_text_tail(meta_path.parent / "log.txt")
                )
                jobs.append(
                    {
                        "id": meta.get("id"),
                        "job_key": meta.get("job_key"),
                        "status": meta.get("status"),
                        "created_at": meta.get("created_at"),
                        "ended_at": meta.get("ended_at"),
                        "remote_url": remote_url,
                    }
                )
        remotes: list[dict[str, Any]] = []
        runs_dir = self.outputs / "runs"
        if runs_dir.exists():
            for cb_path in sorted(runs_dir.glob("*/checkpoint_bucket.json")):
                cb = _read_json(cb_path) or {}
                if cb.get("bucket_url") or cb.get("remote_uri"):
                    remotes.append(
                        {
                            "run_id": cb.get("run_id", cb_path.parent.name),
                            "url": cb.get("bucket_url"),
                            "uri": cb.get("remote_uri"),
                        }
                    )
        return {
            "provenance": "live" if (jobs or remotes) else "committed",
            "jobs": jobs,
            "remotes": remotes,
            "bucket_url": "https://huggingface.co/buckets/TKendrick/OpenUI",
        }

    # ---- system + overview aggregate -----------------------------------------

    def system(self) -> dict[str, Any]:
        return {
            "checkpoint_bucket": "hf://buckets/TKendrick/OpenUI",
            "deployment": self.deployment_state(),
            "outputs_present": self.outputs.exists(),
        }

    def overview(self) -> dict[str, Any]:
        boards = self.scoreboards()
        total = sum(b["count"] for b in boards)
        passed = sum(b["passed"] for b in boards)
        runs = self.runs()
        return {
            "scoreboards": boards,
            "experiment_totals": {"count": total, "passed": passed},
            "runs": runs["runs"][:12],
            "runs_provenance": runs["provenance"],
            "checkpoints": self.checkpoints(),
            "data": self.train_data(),
            "test_data": self.test_data(),
            "annotations": self.annotations_summary(),
            "system": self.system(),
        }
