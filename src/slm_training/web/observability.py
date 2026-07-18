"""Read-only observability facade over the repo's on-disk evidence.

Every dashboard view flows through :class:`Readers`. The methods are pure reads
with tolerant JSON parsing and **cold-start fallback**: because ``outputs/`` is
gitignored and empty on a fresh checkout (and on a read-only Vercel deploy), each
reader falls back to the committed source of truth — ``docs/design/*.json``,
``docs/MODEL_CARD.md``, and ``src/slm_training/resources/``. Every payload carries a ``provenance``
of ``"live"`` (read from ``outputs/``) or ``"committed"`` (fallback snapshot) so
the UI never passes a committed snapshot off as current state — matching the
repo's honesty ethos.

This module must stay import-light (no torch, no model loading) so the Vercel
entrypoint keeps cold-starting.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from functools import cached_property
from pathlib import Path
from typing import Any, Callable

from slm_training.data.store import DataStore
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
)
from slm_training.autoresearch.run_insights import (
    RunInsightSubmission,
    enrich_with_openai,
    load_run_insights,
    save_enrichment,
)
from slm_training.lineage.records import content_sha
from slm_training.lineage.store import LineageStore, utc_now
from slm_training.web.comparisons import BlindedComparisonStore
from slm_training.web.deployments import DeploymentRegistry

# docs/design scoreboard file names by matrix kind.
SCOREBOARD_FILES: dict[str, str] = {
    "quality": "quality-matrix-results.json",
    "grammar": "grammar-matrix-results.json",
    "perf": "perf-matrix-results.json",
    "phase": "phase-abc-results.json",
}
RESEARCH_SCOREBOARD_KIND = "research"

# Suite ordering used across the UI (matches ALLOWED_SPLITS eval order).
SUITE_ORDER = ("smoke", "held_out", "adversarial", "ood", "rico_held")
TRACKS = ("twotower", "causal_lm")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


def _plain_markdown(value: str | None) -> str:
    text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", value or "")
    return text.replace("`", "").replace("**", "").strip()


def _format_parameters(value: int | None, *, estimated: bool = False) -> str:
    if not value:
        return "—"
    prefix = "≈" if estimated else ""
    if value >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{prefix}{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{prefix}{value / 1_000:.0f}K"
    return f"{prefix}{value}"


def _format_bytes(value: int | None, *, estimated: bool = False) -> str:
    if not value:
        return "—"
    prefix = "≈" if estimated else ""
    if value >= 1_000_000_000:
        return f"{prefix}{value / 1_000_000_000:.2f} GB"
    return f"{prefix}{value / 1_000_000:.2f} MB"


_METRIC_LABELS = {
    "meaningful_program_rate": "Meaningful",
    "structural_similarity": "Structure",
    "component_type_recall": "Type recall",
    "placeholder_fidelity": "Fidelity",
    "reward_score": "Reward",
}


def gate_metric_keys() -> list[str]:
    """The aggregate metric levers, in ship-gate policy order.

    The ship-gate policy is the canonical statement of what training is
    optimizing; deriving the dashboard's metric set from it means a policy
    change (adding/dropping a lever) propagates to every metric surface.
    """
    keys: list[str] = []
    for mins in DEFAULT_SHIP_GATES.values():
        for key in mins:
            if key not in keys:
                keys.append(key)
    return keys


def metric_label(key: str) -> str:
    return _METRIC_LABELS.get(key, key.replace("_", " "))


def scoreboard_metric_columns() -> list[dict[str, str]]:
    """Return policy-ordered per-suite columns for experiment scoreboards."""
    return [
        {
            "key": f"{suite}__{metric}",
            "suite": suite,
            "metric": metric,
            "label": f"{suite.replace('_', ' ')} {metric_label(metric).lower()}",
        }
        for suite, minimums in DEFAULT_SHIP_GATES.items()
        for metric in minimums
    ]


def _normalize_metrics(metrics: Any) -> dict[str, float]:
    """Re-key an arbitrary metrics dict onto the ship-gate levers.

    Legacy evaluation reports predate the meaningful/parse split; their
    parse_rate feeds the meaningful lever so cross-era comparisons stay on
    one metric vocabulary.
    """
    if not isinstance(metrics, dict):
        return {}
    out: dict[str, float] = {}
    for key in gate_metric_keys():
        value = metrics.get(key)
        if value is None and key == "meaningful_program_rate":
            value = metrics.get("parse_rate")
        if isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _suite_metrics(suites: Any) -> tuple[dict[str, float], int]:
    """Collapse suite metrics with sample-count weighting for fair comparisons."""
    keys = tuple(gate_metric_keys())
    totals = {key: 0.0 for key in keys}
    weights = {key: 0 for key in keys}
    sample_count = 0
    if not isinstance(suites, dict):
        return {}, 0
    for suite in suites.values():
        if not isinstance(suite, dict):
            continue
        weight = max(1, int(suite.get("n") or 0))
        sample_count += int(suite.get("n") or 0)
        for key in keys:
            value = suite.get(key)
            if value is None and key == "meaningful_program_rate":
                # Legacy scoreboard rows predate the meaningful/parse split.
                value = suite.get("parse_rate")
            if isinstance(value, (int, float)):
                totals[key] += float(value) * weight
                weights[key] += weight
    return (
        {key: totals[key] / weights[key] for key in keys if weights[key]},
        sample_count,
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


class Readers:
    """Facade over the repo's evidence tree. ``root`` is the repo root."""

    def __init__(
        self, root: Path | str = Path("."), *, persist_insights: bool = False
    ) -> None:
        self.root = Path(root)
        self.docs_design = self.root / "docs" / "design"
        self.model_card = self.root / "docs" / "MODEL_CARD.md"
        self.outputs = self.root / "outputs"
        self.fixtures = self.root / "src" / "slm_training" / "resources"
        self.data_store = DataStore(self.root)
        self.published_train_root = self.data_store.published_root / "train"
        self.dashboard_snapshot = self.root / "src" / "slm_training" / "web" / "static" / "dashboard_snapshot.json"
        self.lineage = LineageStore(self.outputs / "lineage")
        self.deployments = DeploymentRegistry(self.outputs / "lineage" / "deployments")
        self.comparisons = BlindedComparisonStore(
            self.data_store.local_root / "annotation" / "comparisons.jsonl"
        )
        self.persist_insights = persist_insights
        # Stat-fingerprint memo: the dashboard polls /api/overview every 15s per
        # client and each request consults the same committed evidence files
        # several times. Values are recomputed whenever any watched file's
        # (mtime_ns, size) changes, so reads stay exactly as fresh as before.
        self._stat_cache: dict[str, tuple[tuple[Any, ...], Any]] = {}

    def _fresh(
        self, name: str, watched: list[Path], compute: Callable[[], Any]
    ) -> Any:
        key: list[tuple[str, int | None, int | None]] = []
        for path in watched:
            try:
                stat = path.stat()
                key.append((str(path), stat.st_mtime_ns, stat.st_size))
            except OSError:
                key.append((str(path), None, None))
        fingerprint = tuple(key)
        hit = self._stat_cache.get(name)
        if hit is not None and hit[0] == fingerprint:
            return hit[1]
        value = compute()
        self._stat_cache[name] = (fingerprint, value)
        return value

    # ---- scoreboards (experiment / perf matrices) -----------------------------

    def _research_results(self) -> list[dict[str, Any]]:
        """Normalize current per-iteration evidence into the dashboard contract."""
        results: list[dict[str, Any]] = []
        for path in self.docs_design.glob("iter-*.json"):
            payload = _read_json(path)
            if not isinstance(payload, dict):
                continue
            evaluation = payload.get("evaluation")
            evaluation = evaluation if isinstance(evaluation, dict) else {}
            suites = payload.get("suites") or evaluation.get("suites")
            if not isinstance(suites, dict):
                continue
            train_result = payload.get("train_result")
            train_result = train_result if isinstance(train_result, dict) else {}
            train = payload.get("train")
            train = train if isinstance(train, dict) else {}
            run_id = (
                payload.get("run_id")
                or train_result.get("run_id")
                or train.get("run_id")
                or evaluation.get("run_id")
            )
            if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
                continue
            gates = payload.get("ship_gates") or evaluation.get("ship_gates")
            gates = gates if isinstance(gates, dict) else {}
            gate_pass = gates.get("pass")
            if not isinstance(gate_pass, bool) and isinstance(
                evaluation.get("failed_gates"), int
            ):
                gate_pass = evaluation["failed_gates"] == 0
            agentv = payload.get("agentv") or evaluation.get("agentv")
            agentv = agentv if isinstance(agentv, dict) else {}
            scoreboard_path = payload.get("scoreboard") or evaluation.get("scoreboard")
            run_dir = (
                str(Path(scoreboard_path).parent)
                if isinstance(scoreboard_path, str)
                else train.get("path")
            )
            results.append(
                {
                    "id": run_id,
                    "run_id": run_id,
                    "description": payload.get("campaign")
                    or payload.get("conclusion")
                    or payload.get("status")
                    or path.stem,
                    "date": payload.get("date_utc") or payload.get("date"),
                    "pass": gate_pass,
                    "suites": suites,
                    "agentv": agentv,
                    "trace_id": train_result.get("trace_id") or train.get("trace_id"),
                    "run_dir": run_dir,
                    "source": f"docs/design/{path.name}",
                }
            )

        def order(row: dict[str, Any]) -> tuple[str, int, str]:
            match = re.search(r"[Ee](\d+)", str(row.get("id") or ""))
            return (
                str(row.get("date") or ""),
                int(match.group(1)) if match else -1,
                str(row.get("id") or ""),
            )

        return sorted(results, key=order, reverse=True)

    def scoreboard(self, kind: str) -> dict[str, Any]:
        if kind == RESEARCH_SCOREBOARD_KIND:
            watched = sorted(self.docs_design.glob("iter-*.json"))
        else:
            filename = SCOREBOARD_FILES.get(kind)
            if filename is None:
                return {"kind": kind, "provenance": "unknown", "results": [], "meta": {}}
            watched = [self.docs_design / filename, self.dashboard_snapshot]
        return self._fresh(
            f"scoreboard:{kind}", watched, lambda: self._scoreboard_uncached(kind)
        )

    def _scoreboard_uncached(self, kind: str) -> dict[str, Any]:
        if kind == RESEARCH_SCOREBOARD_KIND:
            results = self._research_results()
            return {
                "kind": kind,
                "provenance": "committed",
                "reference": "docs/design/iter-*.json",
                "meta": {"latest": results[0].get("date") if results else None},
                "metric_columns": scoreboard_metric_columns(),
                "count": len(results),
                "passed": sum(row.get("pass") is True for row in results),
                "results": results,
            }
        filename = SCOREBOARD_FILES.get(kind)
        if filename is None:
            return {"kind": kind, "provenance": "unknown", "results": [], "meta": {}}
        payload = _read_json(self.docs_design / filename)
        results = _results_of(payload)
        snapshot = _read_json(self.dashboard_snapshot) or {}
        if kind == "quality":
            published = _results_of(snapshot.get("quality_results"))
            known = {r.get("run_id") or r.get("id") for r in results}
            results.extend(r for r in published if (r.get("run_id") or r.get("id")) not in known)
        meta = {k: v for k, v in (payload or {}).items() if k != "results"} if isinstance(
            payload, dict
        ) else {}
        passed = sum(1 for r in results if r.get("pass") is True)
        return {
            "kind": kind,
            "provenance": "committed",
            "reference": f"docs/design/{filename}",
            "meta": meta,
            "metric_columns": scoreboard_metric_columns(),
            "count": len(results),
            "passed": passed,
            "results": results,
        }

    def scoreboards(self) -> list[dict[str, Any]]:
        index: list[dict[str, Any]] = []
        for kind in (*SCOREBOARD_FILES, RESEARCH_SCOREBOARD_KIND):
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
                        "trace_id": manifest.get("trace_id")
                        or (
                            _read_json(
                                self.outputs
                                / "runs"
                                / current.parent.name
                                / "trace.json"
                            )
                            or {}
                        ).get("trace_id"),
                    }
                )
        # Add committed evidence even when lineage exists: most experiment runs are
        # intentionally not promotable and therefore have no lineage manifest.
        snapshot = _read_json(self.dashboard_snapshot) or {}
        derived: list[dict[str, Any]] = []
        for row in _results_of(snapshot.get("runs")):
            derived.append({**row, "provenance": "committed-snapshot"})
        for kind in ("quality", "grammar", "perf", RESEARCH_SCOREBOARD_KIND):
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
                        "provenance": (
                            "live"
                            if self._run_dir(
                                str(row.get("run_id") or row.get("id") or ""), row
                            ).is_dir()
                            else "committed"
                        ),
                    }
                )
        known = {row.get("run_id") for row in live}
        for row in derived:
            if row.get("run_id") not in known:
                live.append(row)
                known.add(row.get("run_id"))
        return {
            "provenance": (
                "live" if any(row.get("provenance") == "live" for row in live) else "committed"
            ),
            "runs": live,
        }

    def _scoreboard_row(self, run_id: str) -> dict[str, Any] | None:
        """Find the matrix row (with suites) whose run_id or experiment id matches."""
        for kind in (RESEARCH_SCOREBOARD_KIND, "quality", "grammar", "perf"):
            for row in self.scoreboard(kind)["results"]:
                if run_id in (row.get("run_id"), row.get("id")):
                    return {"matrix": kind, **row}
        return None

    def _run_dir(self, run_id: str, scoreboard: dict[str, Any] | None = None) -> Path:
        declared = (scoreboard or {}).get("run_dir")
        if isinstance(declared, str):
            candidate = self.root / declared
            if candidate.is_dir():
                return candidate
        # runs() probes this per scoreboard row; try the canonical location
        # with one stat before falling back to the recursive globs.
        direct = self.outputs / "runs" / run_id
        if direct.is_dir():
            return direct
        for candidate in (
            *self.outputs.glob(f"runs/*/{run_id}"),
            *self.outputs.glob(f"autoresearch/*/runs/{run_id}"),
        ):
            if candidate.is_dir():
                return candidate
        return direct

    def run(self, run_id: str) -> dict[str, Any]:
        if not _RUN_ID_RE.fullmatch(run_id):
            return {
                "run_id": run_id,
                "provenance": "missing",
                "manifest": None,
                "scoreboard": None,
                "train_summary": None,
                "gates": None,
                "telemetry": None,
                "matrix_result": None,
                "insights": None,
            }
        scoreboard = self._scoreboard_row(run_id)
        run_dir = self._run_dir(run_id, scoreboard)
        live = {
            "train_summary": _read_json(run_dir / "train_summary.json"),
            "gates": _read_json(run_dir / "gates.json"),
            "telemetry": _read_json(run_dir / "train_telemetry.json"),
            "matrix_result": _read_json(run_dir / "matrix_result.json"),
            "trace": _read_json(run_dir / "trace.json"),
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
        artifacts = dict(live)
        if (
            artifacts["gates"] is None
            and scoreboard is not None
            and "suites" in scoreboard
        ):
            artifacts["gates"] = evaluate_ship_gates(scoreboard["suites"])
        insights = load_run_insights(
            run_dir,
            run_id=run_id,
            scoreboard=scoreboard,
            persist=self.persist_insights,
        )
        return {
            "run_id": run_id,
            "provenance": "live" if has_live else "committed",
            "manifest": lineage_manifest,
            "scoreboard": scoreboard,
            "insights": insights,
            **artifacts,
        }

    def save_run_insights(
        self, run_id: str, submission: RunInsightSubmission
    ) -> dict[str, Any]:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("invalid run id")
        return save_enrichment(
            self._run_dir(run_id, self._scoreboard_row(run_id)),
            run_id=run_id,
            submission=submission,
            scoreboard=self._scoreboard_row(run_id),
        )

    def enrich_run_with_openai(self, run_id: str) -> dict[str, Any]:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError("invalid run id")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OpenAI fallback is not configured")
        scoreboard = self._scoreboard_row(run_id)
        run_dir = self._run_dir(run_id, scoreboard)
        report = load_run_insights(run_dir, run_id=run_id, scoreboard=scoreboard)
        submission = enrich_with_openai(report)
        return save_enrichment(
            run_dir,
            run_id=run_id,
            submission=submission,
            scoreboard=scoreboard,
        )

    def rl_traces(self, run_id: str, *, offset: int = 0, limit: int = 20) -> dict[str, Any]:
        """Read a page of normalized RL traces; raw PyTorch dumps stay remote."""
        if not _RUN_ID_RE.fullmatch(run_id):
            return {
                "run_id": run_id,
                "offset": offset,
                "limit": limit,
                "total": 0,
                "count": 0,
                "invalid_rows": 0,
                "traces": [],
                "provenance": "missing",
            }
        run_dir = self._run_dir(run_id, self._scoreboard_row(run_id))
        trace_ref = _read_json(run_dir / "trace.json") or {}
        bundle_value = str(trace_ref.get("bundle") or "")
        bundle = Path(bundle_value)
        if bundle_value and not bundle.is_absolute():
            bundle = self.root / bundle
        path = (
            bundle / "domain" / "molt" / "rl_traces.jsonl"
            if bundle_value
            else Path("/__missing_trace_bundle__")
        )
        if not path.is_file():
            path = run_dir / "rl_traces.jsonl"
        traces: list[dict[str, Any]] = []
        total = 0
        invalid = 0
        try:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        invalid += 1
                        continue
                    if not isinstance(row, dict) or row.get("run_id") != run_id:
                        invalid += 1
                        continue
                    if offset <= total < offset + limit:
                        traces.append(row)
                    total += 1
        except OSError:
            pass
        return {
            "run_id": run_id,
            "offset": offset,
            "limit": limit,
            "total": total,
            "count": len(traces),
            "invalid_rows": invalid,
            "traces": traces,
            "provenance": "live" if path.exists() else "missing",
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

    @cached_property
    def _comparable_twotower_metrics(self) -> dict[str, Any]:
        checkpoint = self.fixtures / "checkpoints" / "playground_demo" / "last.pt"
        meta = _read_json(checkpoint.with_suffix(".meta.json")) or {}
        parameter_count = meta.get("parameter_count")
        if not isinstance(parameter_count, int):
            parameter_count = None
        checkpoint_bytes = checkpoint.stat().st_size if checkpoint.exists() else None

        perf = _read_json(self.docs_design / "perf-matrix-results.json") or {}
        rates = [
            float(row["tokens_per_sec"])
            for row in _results_of(perf)
            if isinstance(row.get("tokens_per_sec"), (int, float))
        ]
        throughput = f"{min(rates):.0f}–{max(rates):.0f} tok/s CPU" if rates else "—"
        return {
            "parameter_count": parameter_count,
            "checkpoint_bytes": checkpoint_bytes,
            "throughput": throughput,
        }

    @cached_property
    def _perf_throughput(self) -> dict[str, float]:
        return {
            str(row.get("run_id") or row.get("id")): float(row["tokens_per_sec"])
            for row in self.scoreboard("perf")["results"]
            if isinstance(row.get("tokens_per_sec"), (int, float))
        }

    def _checkpoint_resource_metrics(
        self, *, run_id: str, role: str, kind: str, location: str, status: str
    ) -> dict[str, str]:
        evidence = " ".join((role, kind, location, status)).casefold()
        architecture = (
            "Causal LM"
            if "causal_lm" in evidence
            else "TwoTower · frozen HF"
            if any(token in evidence for token in ("hf-context", "smollm", "135m"))
            else "Grammar diffusion"
            if "grammar" in evidence
            else "TwoTower · scratch"
        )

        checkpoint: Path | None = None
        raw_path = _plain_markdown(location).split(" ", 1)[0]
        if raw_path.endswith(".pt"):
            candidate = Path(raw_path)
            candidate = candidate if candidate.is_absolute() else self.root / candidate
            try:
                candidate.relative_to(self.root)
                if candidate.exists():
                    checkpoint = candidate
            except ValueError:
                pass

        parameter_count: int | None = None
        checkpoint_bytes: int | None = None
        throughput: float | None = None
        measured: list[str] = []
        if checkpoint is not None:
            checkpoint_bytes = checkpoint.stat().st_size
            measured.append("checkpoint size")
            meta = _read_json(checkpoint.with_suffix(".meta.json")) or {}
            if isinstance(meta.get("parameter_count"), int):
                parameter_count = meta["parameter_count"]
                measured.append("parameters")
            summary = _read_json(checkpoint.parent.parent / "train_summary.json") or {}
            track = (
                summary.get("track") if isinstance(summary.get("track"), dict) else {}
            )
            counts = [
                track.get("trainable_params"),
                track.get("frozen_params"),
            ]
            if any(isinstance(value, int) for value in counts):
                parameter_count = sum(
                    value for value in counts if isinstance(value, int)
                )
                if "parameters" not in measured:
                    measured.append("parameters")

        if run_id in self._perf_throughput:
            throughput = self._perf_throughput[run_id]
            measured.append("throughput")

        if parameter_count or checkpoint_bytes or throughput is not None:
            comparable = (
                self._comparable_twotower_metrics
                if architecture == "TwoTower · scratch"
                else {}
            )
            return {
                "architecture": architecture,
                "parameters": (
                    _format_parameters(parameter_count)
                    if parameter_count
                    else _format_parameters(
                        comparable.get("parameter_count"), estimated=True
                    )
                    if comparable
                    else "≈135M"
                    if architecture == "TwoTower · frozen HF"
                    else "—"
                ),
                "model_size": (
                    _format_bytes(checkpoint_bytes)
                    if checkpoint_bytes
                    else _format_bytes(
                        comparable.get("checkpoint_bytes"), estimated=True
                    )
                    if comparable
                    else "≈270 MB BF16"
                    if architecture == "TwoTower · frozen HF"
                    else "—"
                ),
                "throughput": (
                    f"{throughput:.1f} tok/s"
                    if throughput is not None
                    else f"≈{comparable['throughput']}"
                    if comparable and comparable.get("throughput") != "—"
                    else "—"
                ),
                "resource_basis": (
                    f"Measured from local {', '.join(measured)}; missing fields "
                    "use the marked comparable architecture estimate."
                ),
            }

        if architecture == "TwoTower · frozen HF":
            return {
                "architecture": architecture,
                "parameters": "≈135M",
                "model_size": "≈270 MB BF16",
                "throughput": "—",
                "resource_basis": (
                    "Architecture estimate for the frozen SmolLM2-135M context; "
                    "checkpoint footprint and throughput were not recorded."
                ),
            }
        if architecture == "TwoTower · scratch":
            comparable = self._comparable_twotower_metrics
            return {
                "architecture": architecture,
                "parameters": _format_parameters(
                    comparable["parameter_count"], estimated=True
                ),
                "model_size": _format_bytes(
                    comparable["checkpoint_bytes"], estimated=True
                ),
                "throughput": (
                    f"≈{comparable['throughput']}"
                    if comparable["throughput"] != "—"
                    else "—"
                ),
                "resource_basis": (
                    "Comparable estimate from the committed scratch TwoTower "
                    "fixture and CPU perf matrix; not measured for this checkpoint."
                ),
            }
        return {
            "architecture": architecture,
            "parameters": "—",
            "model_size": "—",
            "throughput": "—",
            "resource_basis": "No comparable architecture benchmark is recorded.",
        }

    def _with_resource_metrics(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            **row,
            **self._checkpoint_resource_metrics(
                run_id=str(row.get("run_id") or ""),
                role=str(row.get("role") or row.get("track") or ""),
                kind=str(row.get("kind") or ""),
                location=str(row.get("location") or ""),
                status=str(row.get("status") or ""),
            ),
        }

    def checkpoints(
        self, *, deployment: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        roster: list[dict[str, Any]] = []
        card_roster, _ = self._model_card_rows()
        for row in card_roster:
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
        return {
            "checkpoints": [self._with_resource_metrics(row) for row in roster],
            "deployment": deployment
            if deployment is not None
            else self.deployment_state(),
        }

    def gates_for_run(self, run_id: str) -> dict[str, Any]:
        if not _RUN_ID_RE.fullmatch(run_id):
            return {
                "provenance": "missing",
                "policy": DEFAULT_SHIP_GATES,
                "actual": {},
                "gates": {},
                "failures": ["invalid_run_id"],
                "pass": False,
            }
        gates = _read_json(self._run_dir(run_id, self._scoreboard_row(run_id)) / "gates.json")
        if gates is not None:
            return {"provenance": "live", **gates}
        # Cold start: find the run's suites in the committed scoreboards and
        # evaluate the default policy so the UI always has a gate matrix.
        for kind in (RESEARCH_SCOREBOARD_KIND, "quality", "grammar"):
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
        refs = self.data_store.versions("train")
        generated_versions = sorted(
            ref.dataset_id for ref in refs if ref.storage in {"local", "legacy"}
        )
        published_versions = sorted(ref.dataset_id for ref in refs if ref.storage == "git")
        versions = [
            "examples",
            *sorted(set(generated_versions) | set(published_versions)),
        ]
        if version == "examples" or not (generated_versions or published_versions):
            return {
                "provenance": "committed",
                "versions": versions,
                "version": "examples",
                **self._fixture_data(),
            }
        available = set(generated_versions) | set(published_versions)
        if version in available:
            chosen = str(version)
        elif generated_versions:
            chosen = generated_versions[-1]
        elif "remediated_roots_judged" in published_versions:
            chosen = "remediated_roots_judged"
        else:
            chosen = published_versions[-1]
        ref = self.data_store.resolve("train", chosen)
        live = ref.storage in {"local", "legacy"}
        vdir = ref.path
        stats = _read_json(vdir / "stats.json") or {}
        manifest = _read_json(vdir / "manifest.json") or {}
        return {
            "provenance": "live" if live else "committed",
            "storage": ref.storage,
            "path": ref.path.relative_to(self.root).as_posix(),
            "fingerprint": ref.fingerprint,
            "trace_id": manifest.get("trace_id"),
            "versions": versions,
            "version": chosen,
            "stats": stats,
            "source_families": manifest.get("source_family_stats")
            or manifest.get("source_families")
            or {},
            "record_count": stats.get("record_count"),
        }

    def train_records(
        self,
        version: str,
        *,
        split: str | None = None,
        source: str | None = None,
        query: str | None = None,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9._,-]{1,64}", version):
            return {"version": version, "count": 0, "offset": 0, "records": []}
        if version == "examples":
            path = self.fixtures / "train_seeds.jsonl"
        else:
            try:
                path = self.data_store.resolve("train", version).path / "records.jsonl"
            except (FileNotFoundError, ValueError):
                path = Path("/__missing__")
        rows = _read_jsonl(path, limit=None)
        if split:
            rows = [r for r in rows if r.get("split") == split]
        sources = sorted({str(r.get("source") or "unknown") for r in rows})
        if source:
            rows = [r for r in rows if str(r.get("source") or "unknown") == source]
        if query:
            needle = query.casefold()
            rows = [
                r
                for r in rows
                if needle
                in " ".join(
                    str(r.get(key) or "") for key in ("id", "source", "prompt", "openui")
                ).casefold()
            ]
        start = max(0, offset)
        return {
            "version": version,
            "count": len(rows),
            "offset": start,
            "limit": limit,
            "sources": sources,
            "records": rows[start : start + limit],
        }

    def preference_data(self) -> dict[str, Any]:
        """Committed exact-state and pairwise corpora consumed by trainers."""
        root = self.data_store.published_root / "preference"
        rows: list[dict[str, Any]] = []
        if root.is_dir():
            for directory in sorted(path for path in root.iterdir() if path.is_dir()):
                manifest = _read_json(directory / "manifest.json") or {}
                if manifest.get("kind") != "decision_event_corpus":
                    continue
                splits = manifest.get("splits") or {}
                evidence = manifest.get("evidence_kinds") or {}
                counterfactual = int(evidence.get("counterfactual") or 0)
                rows.append(
                    {
                        "dataset_id": manifest.get("dataset_id") or directory.name,
                        "kind": "exact-state decisions",
                        "records": int(manifest.get("record_count") or 0),
                        "train": int(splits.get("train") or 0),
                        "held_out": int(splits.get("held_out") or 0),
                        "evidence": " · ".join(
                            f"{key}:{value}"
                            for key, value in evidence.items()
                            if int(value or 0) > 0
                        ),
                        "usage": (
                            "semantic preference training"
                            if counterfactual
                            else "decoder evidence only"
                        ),
                        "fingerprint": str(
                            manifest.get("content_fingerprint") or ""
                        )[:12],
                    }
                )
        return {"provenance": "committed", "rows": rows}

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
        test_root = self.data_store.local_root / "eval"
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
        snapshot = _read_json(self.dashboard_snapshot) or {}
        snapshot_test = snapshot.get("test_data") if isinstance(snapshot, dict) else None
        if isinstance(snapshot_test, dict):
            return {"provenance": "committed-snapshot", **snapshot_test, "records": None}
        # Cold start: committed fixture test seeds by split.
        rows = _read_jsonl(self.fixtures / "test_seeds.jsonl")
        sizes = {suite: 0 for suite in SUITE_ORDER}
        for r in rows:
            split = r.get("split")
            if split in sizes:
                sizes[split] += 1
        return {"provenance": "committed", "version": None, "suites": sizes}

    def test_records(
        self, suite: str | None = None, *, query: str | None = None,
        offset: int = 0, limit: int = 50,
    ) -> dict[str, Any]:
        snapshot = _read_json(self.dashboard_snapshot) or {}
        data = snapshot.get("test_data") if isinstance(snapshot, dict) else None
        rows = list((data or {}).get("records") or [])
        if suite:
            rows = [r for r in rows if r.get("suite") == suite]
        if query:
            needle = query.casefold()
            rows = [r for r in rows if needle in " ".join(str(r.get(k) or "") for k in ("id", "suite", "prompt", "openui")).casefold()]
        start = max(0, offset)
        return {"provenance": "committed-snapshot", "version": (data or {}).get("version"), "count": len(rows), "offset": start, "limit": limit, "records": rows[start:start + limit]}

    # ---- annotations / comparisons -------------------------------------------

    def annotations_summary(self) -> dict[str, Any]:
        ann = self.data_store.local_root / "annotation"
        prefs = self.data_store.local_root / "preference"
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

    # ---- reference model + persisted performance insights -------------------

    def _model_card_rows(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        return self._fresh(
            "model_card_rows", [self.model_card], self._model_card_rows_uncached
        )

    def _model_card_rows_uncached(
        self,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        try:
            text = self.model_card.read_text(encoding="utf-8")
        except OSError:
            text = ""
        roster = _parse_markdown_table(text, "## Current checkpoint roster")
        history = _parse_markdown_table(text, "## Checkpoint history")
        return roster, history

    def _reference_models(self) -> tuple[list[dict[str, Any]], str]:
        roster, history = self._model_card_rows()
        references: list[dict[str, Any]] = []
        champion_identity: list[dict[str, Any]] = []

        for track in TRACKS:
            try:
                pointer = self.lineage.champion(track)
            except (OSError, ValueError):
                pointer = None
            if pointer is None:
                continue
            metrics: dict[str, float] = {}
            suite_sizes: dict[str, int] = {}
            try:
                report = self.lineage.load_report(pointer.evaluation_report_sha)
                metrics = {
                    key: float(value)
                    for key, value in report.metrics.items()
                    if key
                    in {
                        "parse_rate",
                        "placeholder_fidelity",
                        "structural_similarity",
                        "reward_score",
                    }
                    and isinstance(value, (int, float))
                }
                suite_sizes = dict(report.suite_sizes)
            except (FileNotFoundError, OSError, ValueError, TypeError):
                pass
            reference = {
                "role": "Champion",
                "track": track,
                "run_id": pointer.run_id,
                "kind": "lineage champion",
                "location": pointer.artifact_uri,
                "status": "champion",
                "evaluation_status": "evaluated" if metrics else "evaluation unavailable",
                "metrics": metrics,
                "suite_sizes": suite_sizes,
                "created_at": pointer.created_at,
                "provenance": "live",
            }
            references.append(reference)
            champion_identity.append(pointer.to_dict())

        # A model-card champion remains visible until it is represented by a live
        # lineage pointer. This preserves the repo's existing scratch-champion state.
        live_run_ids = {row["run_id"] for row in references}
        for row in roster:
            role = _plain_markdown(row.get("role"))
            run_id = _plain_markdown(row.get("run id"))
            if "champion" not in role.casefold() or run_id in live_run_ids:
                continue
            references.append(
                {
                    "role": role,
                    "track": "twotower",
                    "run_id": run_id,
                    "kind": _plain_markdown(row.get("kind")),
                    "location": _plain_markdown(row.get("location")),
                    "status": _plain_markdown(row.get("status")),
                    "evaluation_status": "evaluation not attached",
                    "metrics": {},
                    "suite_sizes": {},
                    "created_at": None,
                    "provenance": "committed",
                }
            )

        # The final history row is the newest checkpoint recorded by the canonical
        # model card. Keep it beside champions even when it has not been promoted.
        if history:
            newest = history[-1]
            run_id = _plain_markdown(newest.get("run id"))
            matched = next(
                (
                    row
                    for row in roster
                    if _plain_markdown(row.get("run id")) == run_id
                ),
                {},
            )
            if run_id and run_id not in {row["run_id"] for row in references}:
                scoreboard = self._scoreboard_row(run_id)
                metrics, suite_n = _suite_metrics((scoreboard or {}).get("suites"))
                references.append(
                    {
                        "role": "Latest checkpoint",
                        "track": "twotower",
                        "run_id": run_id,
                        "kind": _plain_markdown(matched.get("kind")) or "checkpoint",
                        "location": _plain_markdown(newest.get("bucket / path"))
                        or _plain_markdown(matched.get("location")),
                        "status": _plain_markdown(matched.get("status"))
                        or _plain_markdown(newest.get("notes")),
                        "evaluation_status": "evaluated" if metrics else "not evaluated",
                        "metrics": metrics,
                        "suite_sizes": {"all": suite_n} if suite_n else {},
                        "created_at": _plain_markdown(newest.get("date (utc)")),
                        "provenance": "committed",
                    }
                )

        if not references and roster:
            row = roster[0]
            references.append(
                {
                    "role": _plain_markdown(row.get("role")),
                    "track": "twotower",
                    "run_id": _plain_markdown(row.get("run id")),
                    "kind": _plain_markdown(row.get("kind")),
                    "location": _plain_markdown(row.get("location")),
                    "status": _plain_markdown(row.get("status")),
                    "evaluation_status": "evaluation not attached",
                    "metrics": {},
                    "suite_sizes": {},
                    "created_at": None,
                    "provenance": "committed",
                }
            )

        identity = content_sha(
            {
                "champions": champion_identity,
                "model_card_roster": [
                    {
                        "role": _plain_markdown(row.get("role")),
                        "run_id": _plain_markdown(row.get("run id")),
                        "location": _plain_markdown(row.get("location")),
                        "status": _plain_markdown(row.get("status")),
                    }
                    for row in roster
                ],
                "latest_checkpoint_history": (
                    {
                        key: _plain_markdown(value)
                        for key, value in history[-1].items()
                    }
                    if history
                    else None
                ),
            }
        )
        return [self._with_resource_metrics(row) for row in references], identity

    def _performance_rows(
        self, references: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        primary = next(
            (r for r in references if r["track"] == "twotower" and r["metrics"]),
            None,
        )
        # Normalize the reference onto the gate levers so deltas never average
        # different metric vocabularies (e.g. legacy parse_rate-era reports).
        if primary is not None:
            primary = {**primary, "metrics": _normalize_metrics(primary["metrics"])}
            if not primary["metrics"]:
                primary = None
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for kind in (RESEARCH_SCOREBOARD_KIND, "quality", "grammar"):
            for row in self.scoreboard(kind)["results"]:
                run_id = str(row.get("run_id") or row.get("id") or "")
                if not run_id or run_id in seen:
                    continue
                seen.add(run_id)
                metrics, sample_count = _suite_metrics(row.get("suites"))
                if not metrics:
                    continue
                score = sum(metrics.values()) / len(metrics)
                # Deltas compare means over the levers BOTH sides report, so a
                # row missing a lever is never compared across dimensions.
                delta = None
                if primary is not None:
                    common = [
                        key
                        for key in gate_metric_keys()
                        if key in metrics and key in primary["metrics"]
                    ]
                    if common:
                        delta = sum(metrics[k] for k in common) / len(common) - sum(
                            primary["metrics"][k] for k in common
                        ) / len(common)
                rows.append(
                    {
                        "id": row.get("id") or row.get("run_id"),
                        "run_id": row.get("run_id") or row.get("id"),
                        "matrix": kind,
                        "description": row.get("description") or "",
                        "gate_status": (
                            "pass"
                            if row.get("pass") is True
                            else "fail"
                            if row.get("pass") is False
                            else "not recorded"
                        ),
                        "metrics": metrics,
                        "score": score,
                        "sample_count": sample_count,
                        "vs_reference": (
                            f"{delta * 100:+.1f} pp"
                            if delta is not None
                            else "reference not evaluated"
                        ),
                    }
                )
        return rows, primary

    @staticmethod
    def _generate_insights(
        references: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        primary: dict[str, Any] | None,
    ) -> dict[str, list[dict[str, str]]]:
        improvements: list[dict[str, str]] = []
        carry_forward: list[dict[str, str]] = []
        novel: list[dict[str, str]] = []

        if primary is None:
            improvements.append(
                {
                    "finding": "The current reference has no attached standard evaluation, so honest deltas cannot be calculated.",
                    "suggestion": "Run the full standard suites with ship gates and AgentV, then attach that report to the checkpoint or champion.",
                }
            )
        else:
            weakest = min(primary["metrics"], key=primary["metrics"].get)
            improvements.append(
                {
                    "finding": f"The reference's weakest aggregate metric is {weakest.replace('_', ' ')} ({primary['metrics'][weakest]:.1%}).",
                    "suggestion": "Prioritize the failing suite slices for that metric; keep the existing gates fixed while testing the mitigation.",
                }
            )

        failed = [row for row in rows if row["gate_status"] == "fail"]
        if failed:
            improvements.append(
                {
                    "finding": f"{len(failed)} evaluated experiment{'s' if len(failed) != 1 else ''} failed {'their' if len(failed) != 1 else 'its'} recorded gate policy.",
                    "suggestion": "Cluster failures by suite and metric, then branch one mitigation at a time from the current reference.",
                }
            )

        if rows:
            best = max(rows, key=lambda row: row["score"])
            carry_forward.append(
                {
                    "finding": f"{best['id']} has the strongest aggregate evidence ({best['score']:.1%}) across {best['sample_count']} suite examples.",
                    "suggestion": "Carry its recipe forward as the experimental control; change one lever per branch.",
                }
            )
            passing = [row for row in rows if row["gate_status"] == "pass"]
            if passing:
                carry_forward.append(
                    {
                        "finding": f"{len(passing)} experiment{'s' if len(passing) != 1 else ''} cleared the recorded gates.",
                        "suggestion": "Preserve their honesty, constrained-decode, and evaluation settings in descendant recipes.",
                    }
                )

            signatures: dict[tuple[float | None, ...], list[str]] = {}
            for row in rows:
                signature = tuple(
                    round(row["metrics"][key], 6)
                    if isinstance(row["metrics"].get(key), float)
                    else None
                    for key in gate_metric_keys()
                )
                signatures.setdefault(signature, []).append(str(row["id"]))
            same = max(signatures.values(), key=len)
            if len(same) > 1:
                novel.append(
                    {
                        "finding": f"{len(same)} experiments ({', '.join(same)}) have indistinguishable aggregate quality metrics.",
                        "suggestion": "Verify that each lever activates in telemetry; if it does, enlarge the discriminating suites before claiming a gain.",
                    }
                )

            max_examples = max(row["sample_count"] for row in rows)
            if max_examples < 1500:
                novel.append(
                    {
                        "finding": f"The broadest recorded comparison covers only {max_examples} suite examples, below the full RICO ship bar.",
                        "suggestion": "Treat rankings as directional and rerun the finalist on full rico_held before promotion.",
                    }
                )

        if len(references) > 1:
            novel.append(
                {
                    "finding": f"The model card currently exposes {len(references)} reference artifacts, not one universal champion.",
                    "suggestion": "Keep track-specific champions separate and label which reference each future matrix branches from.",
                }
            )

        return {
            "improvements": improvements,
            "carry_forward": carry_forward,
            "novel": novel,
        }

    def performance_insights(self) -> dict[str, Any]:
        references, reference_identity = self._reference_models()
        # The insight cache regenerates when the roster/champions change (by
        # design) — and also when the gate policy changes, since every finding
        # is derived from the policy's metric levers.
        fingerprint = content_sha(
            {"references": reference_identity, "gate_policy": DEFAULT_SHIP_GATES}
        )
        rows, primary = self._performance_rows(references)
        cache_path = self.outputs / "dashboard" / "overview-insights.json"
        cached_payload = _read_json(cache_path) or {}
        cached = (
            cached_payload.get("reference_fingerprint") == fingerprint
            and isinstance(cached_payload.get("insights"), dict)
        )
        persisted = cached
        if cached:
            insights = cached_payload["insights"]
            generated_at = cached_payload.get("generated_at")
        else:
            insights = self._generate_insights(references, rows, primary)
            generated_at = utc_now()
            payload = {
                "reference_fingerprint": fingerprint,
                "reference_run_ids": [row["run_id"] for row in references],
                "generated_at": generated_at,
                "insights": insights,
            }
            try:
                _atomic_json(cache_path, payload)
                persisted = True
            except OSError:
                persisted = False

        passing = sum(row["gate_status"] == "pass" for row in rows)
        basis = (
            f"All deltas use {primary['run_id']} as the TwoTower reference."
            if primary
            else "Reference evaluation is missing; absolute experiment scores are shown and deltas are withheld."
        )
        return {
            "references": references,
            "reference_fingerprint": fingerprint,
            "reference_provenance": (
                "live" if any(row["provenance"] == "live" for row in references) else "committed"
            ),
            "comparison_basis": basis,
            "comparisons": rows,
            "metric_columns": [
                {"key": key, "label": metric_label(key)} for key in gate_metric_keys()
            ],
            "stats": {
                "reference_models": len(references),
                "experiments": len(rows),
                "passing": passing,
                "comparable": len(rows) if primary else 0,
            },
            "insights": insights,
            "cache": {
                "cached": cached,
                "persisted": persisted,
                "generated_at": generated_at,
                "path": str(cache_path.relative_to(self.root)),
            },
        }

    # ---- system + overview aggregate -----------------------------------------

    def system(self, *, deployment: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            # An existing-but-empty outputs/ is still a cold start.
            outputs_present = self.outputs.exists() and any(self.outputs.iterdir())
        except OSError:
            outputs_present = False
        return {
            "checkpoint_bucket": "hf://buckets/TKendrick/OpenUI",
            "deployment": deployment
            if deployment is not None
            else self.deployment_state(),
            "outputs_present": outputs_present,
        }

    def overview(self) -> dict[str, Any]:
        boards = self.scoreboards()
        total = sum(b["count"] for b in boards)
        passed = sum(b["passed"] for b in boards)
        runs = self.runs()
        # Deployment state is embedded by two sub-views; read it once per request.
        deployment = self.deployment_state()
        return {
            "scoreboards": boards,
            "experiment_totals": {"count": total, "passed": passed},
            "runs": runs["runs"][:12],
            "runs_provenance": runs["provenance"],
            "checkpoints": self.checkpoints(deployment=deployment),
            "data": self.train_data(),
            "test_data": self.test_data(),
            "annotations": self.annotations_summary(),
            "system": self.system(deployment=deployment),
            "performance": self.performance_insights(),
        }
