"""Training-data build pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.train_data.synth import PromptSynthesizer, get_synthesizer


@dataclass
class TrainDataConfig:
    seed_path: Path
    output_root: Path = Path("outputs/train_data")
    version: str = "v0"
    synthesizer: str = "template"
    require_split: str = "train"

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.version


def _fingerprint(openui: str, prompt: str) -> str:
    payload = (prompt.strip() + "\n" + openui.strip()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_record(record: ExampleRecord) -> ExampleRecord:
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


def build_train_data(
    config: TrainDataConfig,
    synthesizer: PromptSynthesizer | None = None,
) -> dict:
    """Load seeds, synthesize, validate, dedupe, and write versioned artifacts."""
    seeds = load_jsonl(config.seed_path)
    synth = synthesizer or get_synthesizer(config.synthesizer)

    collected: list[ExampleRecord] = []
    errors: list[dict] = []

    for seed in seeds:
        if seed.split != config.require_split:
            errors.append(
                {
                    "id": seed.id,
                    "error": f"expected split {config.require_split!r}, got {seed.split!r}",
                }
            )
            continue
        candidates = [seed, *synth.expand(seed)]
        for candidate in candidates:
            try:
                collected.append(_normalize_record(candidate))
            except (ParseError, ValueError) as exc:
                errors.append({"id": candidate.id, "error": str(exc)})

    # Dedupe by (prompt, openui) fingerprint; keep first id.
    deduped: list[ExampleRecord] = []
    seen: set[str] = set()
    for record in collected:
        fp = _fingerprint(record.openui, record.prompt)
        if fp in seen:
            continue
        seen.add(fp)
        deduped.append(record)

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    write_jsonl(records_path, deduped)

    stats = {
        "version": config.version,
        "seed_path": str(config.seed_path),
        "seed_count": len(seeds),
        "collected_count": len(collected),
        "record_count": len(deduped),
        "error_count": len(errors),
        "errors": errors,
        "synthesizer": config.synthesizer,
        "placeholder_vocab_size": len(
            {p for r in deduped for p in r.placeholders}
        ),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "version": config.version,
        "kind": "train_data",
        "records": str(records_path.as_posix()),
        "stats": str(stats_path.as_posix()),
        "record_count": len(deduped),
        "ids": [r.id for r in deduped],
        "built_at": stats["built_at"],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "manifest": manifest,
        "stats": stats,
    }
