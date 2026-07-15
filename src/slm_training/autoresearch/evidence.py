"""Normalize prior runs, telemetry, feedback, data, and research lineage."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from slm_training.autoresearch.schemas import EvidenceItem, EvidenceSnapshot
from slm_training.lineage.records import content_sha

LINEAGE_DOCS = (
    Path("docs/design/research-lineage.md"),
    Path("docs/design/quality-experiment-matrix.md"),
    Path("docs/design/perf-experiment-matrix.md"),
    Path("docs/design/nemo-rl-autoresearch.md"),
)
SUPPORTED_SUFFIXES = {".json", ".jsonl", ".md", ".tsv"}


def collect_evidence(
    roots: Iterable[Path | str],
    *,
    repo_root: Path | str = Path("."),
    max_files: int = 5000,
    max_file_bytes: int = 5_000_000,
) -> EvidenceSnapshot:
    repo_root = Path(repo_root).resolve()
    requested = tuple(str(Path(root)) for root in roots)
    candidates: set[Path] = set()
    for relative in LINEAGE_DOCS:
        path = repo_root / relative
        if path.is_file():
            candidates.add(path)
    for raw in requested:
        root = Path(raw)
        root = root if root.is_absolute() else repo_root / root
        if root.is_file():
            candidates.add(root)
        elif root.is_dir():
            candidates.update(
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
    items: list[EvidenceItem] = []
    # Run diagnoses should reach the bounded researcher prompt before bulk outputs.
    ordered = sorted(
        candidates,
        key=lambda path: (0 if path.name == "run_insights.json" else 1, str(path)),
    )
    for path in ordered[:max_files]:
        size = path.stat().st_size
        if size > max_file_bytes or _looks_sensitive(path):
            continue
        raw = path.read_bytes()
        kind = classify_evidence(path)
        summary, metrics = summarize_evidence(path, raw)
        try:
            shown = str(path.relative_to(repo_root))
        except ValueError:
            shown = str(path)
        items.append(
            EvidenceItem(
                path=shown,
                kind=kind,
                sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=size,
                summary=summary,
                metrics=metrics,
            )
        )
    counts = Counter(item.kind for item in items)
    prior = sorted(
        {
            parts[parts.index("autoresearch") + 1]
            for item in items
            if "autoresearch" in (parts := Path(item.path).parts)
            and parts.index("autoresearch") + 1 < len(parts)
        }
    )
    identity = {
        "roots": requested,
        "items": [item.model_dump(mode="json") for item in items],
    }
    return EvidenceSnapshot(
        snapshot_id=f"evidence-{content_sha(identity)[:16]}",
        roots=requested,
        items=tuple(items),
        source_counts=dict(sorted(counts.items())),
        prior_campaign_ids=tuple(prior),
    )


def classify_evidence(path: Path) -> str:
    name = path.name.lower()
    joined = "/".join(path.parts).lower()
    if name == "run_insights.json":
        return "run_insight"
    if "research-lineage" in name or "experiment-matrix" in name:
        return "repo_lineage"
    if "agentv" in joined or name.endswith(".eval.jsonl"):
        return "agentv"
    if "telemetry" in name or "profile" in name or "bench" in name:
        return "telemetry"
    if "annotation" in joined or "feedback" in joined or "pair" in name:
        return "feedback"
    if "train_data" in joined or "mixture" in name or "manifest" in name:
        return "data_snapshot"
    if "autoresearch" in joined:
        return "prior_campaign"
    if "gate" in name or "eval" in joined or "score" in name:
        return "evaluation"
    if "summary" in name or "runs" in joined or "lineage" in joined:
        return "prior_run"
    return "artifact"


def summarize_evidence(path: Path, raw: bytes) -> tuple[str, dict[str, float]]:
    text = raw.decode("utf-8", errors="replace")
    metrics: dict[str, float] = {}
    if path.suffix == ".json":
        try:
            value = json.loads(text)
            _collect_metrics(value, metrics)
            if path.name == "run_insights.json" and isinstance(value, dict):
                lines = []
                for item in value.get("insights") or []:
                    if isinstance(item, dict):
                        lines.append(
                            f"finding={item.get('finding', '')} suggestion={item.get('suggestion', '')}"
                        )
                generated = (value.get("enrichment") or {}).get("generated") or {}
                if isinstance(generated, dict):
                    if generated.get("summary"):
                        lines.append(f"enriched_summary={generated['summary']}")
                    for cause in generated.get("causes") or []:
                        if isinstance(cause, dict):
                            lines.append(
                                f"hypothesis={cause.get('title', '')}: {cause.get('rationale', '')} suggestion={cause.get('suggestion', '')}"
                            )
                return " | ".join(lines)[:6000], dict(sorted(metrics.items())[:100])
            summary = _json_summary(value)
            return summary[:1500], dict(sorted(metrics.items())[:100])
        except json.JSONDecodeError:
            pass
    if path.suffix == ".jsonl":
        rows = []
        for line in text.splitlines()[:200]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        for row in rows:
            _collect_metrics(row, metrics)
        return f"jsonl rows sampled={len(rows)}", dict(sorted(metrics.items())[:100])
    clean = " ".join(line.strip() for line in text.splitlines() if line.strip())
    return clean[:6000], metrics


def _collect_metrics(value: Any, out: dict[str, float], prefix: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}".strip(".")
            if isinstance(child, (int, float)) and not isinstance(child, bool):
                out[name] = float(child)
            elif len(out) < 250:
                _collect_metrics(child, out, name)
    elif isinstance(value, list):
        for child in value[:20]:
            if len(out) >= 250:
                break
            _collect_metrics(child, out, prefix)


def _json_summary(value: Any) -> str:
    if isinstance(value, dict):
        return "json keys=" + ",".join(sorted(map(str, value))[:40])
    if isinstance(value, list):
        return f"json list rows={len(value)}"
    return f"json {type(value).__name__}"


def _looks_sensitive(path: Path) -> bool:
    lower = path.name.lower()
    return lower.startswith(".env") or any(
        marker in lower for marker in ("token", "credential", "secret")
    )
