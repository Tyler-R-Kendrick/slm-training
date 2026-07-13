"""Testing-data build pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl

DEFAULT_SUITES = ("smoke", "held_out", "adversarial", "ood")


@dataclass
class TestDataConfig:
    seed_path: Path
    output_root: Path = Path("outputs/test_data")
    version: str = "v0"
    suites: tuple[str, ...] = DEFAULT_SUITES
    train_manifest: Path | None = None

    # Prevent pytest from collecting this dataclass as a test class.
    __test__ = False

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.version


def _load_train_ids(manifest_path: Path | None) -> set[str]:
    if manifest_path is None:
        return set()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return set(data.get("ids") or [])


def _normalize(record: ExampleRecord) -> ExampleRecord:
    program = validate(record.openui)
    placeholders = extract_placeholders(record.openui) or list(program.placeholders)
    return ExampleRecord(
        id=record.id,
        prompt=record.prompt.strip(),
        openui=record.openui.strip(),
        placeholders=placeholders,
        split=record.split,
        source=record.source,
        meta=dict(record.meta),
    )


def build_test_data(config: TestDataConfig) -> dict:
    seeds = load_jsonl(config.seed_path)
    train_ids = _load_train_ids(config.train_manifest)

    by_suite: dict[str, list[ExampleRecord]] = {s: [] for s in config.suites}
    errors: list[dict] = []
    leakage: list[str] = []

    for seed in seeds:
        suite = str(seed.meta.get("suite") or seed.split)
        if suite not in by_suite:
            # Skip suites not requested
            continue
        if seed.id in train_ids:
            leakage.append(seed.id)
            continue
        try:
            by_suite[suite].append(_normalize(seed))
        except (ParseError, ValueError) as exc:
            errors.append({"id": seed.id, "error": str(exc)})

    if leakage:
        raise ValueError(
            "test ids overlap train manifest: " + ", ".join(sorted(leakage))
        )

    out_dir = config.output_dir
    suites_dir = out_dir / "suites"
    suites_dir.mkdir(parents=True, exist_ok=True)

    suite_paths: dict[str, str] = {}
    suite_counts: dict[str, int] = {}
    all_ids: list[str] = []

    for suite, records in by_suite.items():
        path = suites_dir / suite / "records.jsonl"
        write_jsonl(path, records)
        suite_paths[suite] = str(path.as_posix())
        suite_counts[suite] = len(records)
        all_ids.extend(r.id for r in records)

    built_at = datetime.now(timezone.utc).isoformat()
    stats = {
        "version": config.version,
        "seed_path": str(config.seed_path),
        "suite_counts": suite_counts,
        "total_records": sum(suite_counts.values()),
        "error_count": len(errors),
        "errors": errors,
        "train_manifest": str(config.train_manifest) if config.train_manifest else None,
        "built_at": built_at,
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "version": config.version,
        "kind": "test_data",
        "suites": suite_paths,
        "stats": str(stats_path.as_posix()),
        "ids": all_ids,
        "suite_counts": suite_counts,
        "built_at": built_at,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "manifest": manifest,
        "stats": stats,
    }
