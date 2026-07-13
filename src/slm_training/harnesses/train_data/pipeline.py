"""Training-data build pipeline (RICO-first)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from slm_training.data.leakage import (
    fingerprint_design_md,
    fingerprint_openui,
    fingerprint_pair,
    fingerprint_prompt,
)
from slm_training.data.rico import load_rico_screens, screen_to_record
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.train_data.synth import PromptSynthesizer, get_synthesizer


@dataclass
class TrainDataConfig:
    seed_path: Path | None = None
    rico_path: Path | None = Path("fixtures/rico/semantic_train.jsonl")
    # rico | fixture | both
    source: str = "rico"
    output_root: Path = Path("outputs/train_data")
    version: str = "v0"
    synthesizer: str = "template"
    require_split: str = "train"
    rico_hf_split: str | None = None
    rico_limit: int | None = None
    max_children: int = 6

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.version


def _normalize_record(record: ExampleRecord) -> ExampleRecord:
    program = validate(record.openui)
    placeholders = list(program.placeholders) or extract_placeholders(record.openui)
    openui = program.serialized or record.openui.strip()
    out = ExampleRecord(
        id=record.id,
        prompt=record.prompt.strip(),
        openui=openui,
        placeholders=placeholders,
        split=record.split,
        source=record.source,
        meta={**dict(record.meta), "parser": "openuidev/lang-core"},
        design_md=record.design_md,
    )
    try:
        from slm_training.design_md import attach_default_design_md

        out = attach_default_design_md(out)
    except Exception:  # noqa: BLE001
        pass
    return out


def _records_from_fixtures(config: TrainDataConfig) -> tuple[list[ExampleRecord], list[dict]]:
    if config.seed_path is None:
        return [], []
    seeds = load_jsonl(config.seed_path)
    errors: list[dict] = []
    out: list[ExampleRecord] = []
    for seed in seeds:
        if seed.split != config.require_split:
            errors.append(
                {
                    "id": seed.id,
                    "error": f"expected split {config.require_split!r}, got {seed.split!r}",
                }
            )
            continue
        out.append(seed)
    return out, errors


def _records_from_rico(config: TrainDataConfig) -> tuple[list[ExampleRecord], list[dict]]:
    if config.rico_path is None and config.rico_hf_split is None:
        return [], [{"error": "rico source selected but no rico_path / rico_hf_split"}]
    screens = load_rico_screens(
        path=config.rico_path,
        hf_split=config.rico_hf_split,
        limit=config.rico_limit,
    )
    out: list[ExampleRecord] = []
    errors: list[dict] = []
    for screen in screens:
        try:
            out.append(
                screen_to_record(
                    screen,
                    split="train",
                    max_children=config.max_children,
                )
            )
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(
                {
                    "id": f"rico_{screen.get('split_src')}_{screen.get('screen_index')}",
                    "error": str(exc),
                }
            )
    return out, errors


def _records_from_awwwards(config: TrainDataConfig) -> tuple[list[ExampleRecord], list[dict]]:
    from slm_training.data.awwwards import AwwwardsConfig, build_awwwards_records

    try:
        records = build_awwwards_records(
            AwwwardsConfig(
                fixture_path=Path("fixtures/awwwards/sites.jsonl"),
                max_sites=config.rico_limit or 20,
            )
        )
        return records, []
    except Exception as exc:  # noqa: BLE001
        return [], [{"error": f"awwwards: {exc}"}]


def build_train_data(
    config: TrainDataConfig,
    synthesizer: PromptSynthesizer | None = None,
) -> dict:
    """Load RICO/fixtures/awwwards, synthesize, validate, dedupe, and write artifacts."""
    source = (config.source or "rico").lower()
    seeds: list[ExampleRecord] = []
    errors: list[dict] = []

    if source in {"fixture", "both", "fixtures", "all"}:
        fixture_records, fixture_errors = _records_from_fixtures(config)
        seeds.extend(fixture_records)
        errors.extend(fixture_errors)
    if source in {"rico", "both", "rico+awwwards", "all"}:
        rico_records, rico_errors = _records_from_rico(config)
        seeds.extend(rico_records)
        errors.extend(rico_errors)
    if source in {"awwwards", "rico+awwwards", "all"}:
        aww_records, aww_errors = _records_from_awwwards(config)
        seeds.extend(aww_records)
        errors.extend(aww_errors)
    allowed = {
        "rico",
        "fixture",
        "fixtures",
        "both",
        "awwwards",
        "rico+awwwards",
        "all",
    }
    if source not in allowed:
        raise ValueError(f"unknown train source {config.source!r}")

    synth = synthesizer or get_synthesizer(config.synthesizer)
    collected: list[ExampleRecord] = []
    for seed in seeds:
        candidates = [seed, *synth.expand(seed)]
        for candidate in candidates:
            try:
                collected.append(_normalize_record(candidate))
            except (ParseError, ValueError) as exc:
                errors.append({"id": candidate.id, "error": str(exc)})

    deduped: list[ExampleRecord] = []
    seen_pairs: set[str] = set()
    prompt_fps: set[str] = set()
    openui_fps: set[str] = set()
    design_md_fps: set[str] = set()
    for record in collected:
        pair = fingerprint_pair(record.prompt, record.openui)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        prompt_fps.add(fingerprint_prompt(record.prompt))
        openui_fps.add(fingerprint_openui(record.openui))
        dm = fingerprint_design_md(record.design_md)
        if dm:
            design_md_fps.add(dm)
        deduped.append(record)

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    write_jsonl(records_path, deduped)

    stats = {
        "version": config.version,
        "source": source,
        "seed_path": str(config.seed_path) if config.seed_path else None,
        "rico_path": str(config.rico_path) if config.rico_path else None,
        "rico_hf_split": config.rico_hf_split,
        "seed_count": len(seeds),
        "collected_count": len(collected),
        "record_count": len(deduped),
        "error_count": len(errors),
        "errors": errors[:50],
        "synthesizer": config.synthesizer,
        "placeholder_vocab_size": len(
            {p for r in deduped for p in r.placeholders}
        ),
        "with_design_md": sum(1 for r in deduped if r.design_md),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "version": config.version,
        "kind": "train_data",
        "source": source,
        "records": str(records_path.as_posix()),
        "stats": str(stats_path.as_posix()),
        "record_count": len(deduped),
        "ids": [r.id for r in deduped],
        "prompt_fingerprints": sorted(prompt_fps),
        "openui_fingerprints": sorted(openui_fps),
        "pair_fingerprints": sorted(seen_pairs),
        "design_md_fingerprints": sorted(design_md_fps),
        "built_at": stats["built_at"],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "manifest": manifest,
        "stats": stats,
    }
