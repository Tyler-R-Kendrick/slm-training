"""Testing-data build pipeline with strict train leakage checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from slm_training.data.leakage import find_leakage, load_train_fingerprints
from slm_training.data.rico import load_rico_screens, screen_to_record
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl

DEFAULT_SUITES = ("smoke", "held_out", "adversarial", "ood", "rico_held")


@dataclass
class TestDataConfig:
    seed_path: Path | None = Path("fixtures/test_seeds.jsonl")
    rico_path: Path | None = Path("fixtures/rico/semantic_test.jsonl")
    # fixture | rico | both
    source: str = "both"
    output_root: Path = Path("outputs/test_data")
    version: str = "v0"
    suites: tuple[str, ...] = DEFAULT_SUITES
    train_manifest: Path | None = None
    require_train_manifest: bool = True
    rico_hf_split: str | None = None
    rico_limit: int | None = None
    rico_hf_cache_path: Path | None = Path("fixtures/rico/hf_test_cache.jsonl")
    max_children: int = 6
    # Map RICO screens into these suites (round-robin).
    # Smoke/held_out stay fixture-curated; RICO uses rico_held.
    rico_suites: tuple[str, ...] = ("rico_held",)
    # Keep pulling / converting until at least this many non-leaking records
    # are kept (None disables). Useful for large HF expansions.
    target_records: int | None = None

    # Prevent pytest from collecting this dataclass as a test class.
    __test__ = False

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.version


def _normalize(record: ExampleRecord) -> ExampleRecord:
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


def _fixture_seeds(config: TestDataConfig) -> list[ExampleRecord]:
    if config.seed_path is None:
        return []
    return load_jsonl(config.seed_path)


def _rico_seeds(config: TestDataConfig) -> list[ExampleRecord]:
    if config.rico_path is None and config.rico_hf_split is None:
        return []
    # Over-fetch when targeting a kept-count so leakage/errors don't undershoot.
    pull_limit = config.rico_limit
    if config.target_records is not None:
        # ~70% post-leakage yield observed on HF test; pull with headroom.
        need = int(config.target_records * 1.6) + 50
        pull_limit = max(pull_limit or 0, need)

    suites = [s for s in config.rico_suites if s in config.suites] or ["held_out"]
    out: list[ExampleRecord] = []

    def _append(screens: list[dict], *, id_prefix: str) -> None:
        for i, screen in enumerate(screens):
            suite = suites[len(out) % len(suites)]
            out.append(
                screen_to_record(
                    screen,
                    split=suite,
                    suite=suite,
                    max_children=config.max_children,
                    id_prefix=id_prefix,
                )
            )

    # Local fixture slice (small, stable).
    if config.rico_path is not None:
        local = load_rico_screens(path=config.rico_path, hf_split=None, limit=None)
        _append(local, id_prefix="rico_eval")

    # Live / cached HF pull — separate id prefix avoids screen_index collisions.
    if config.rico_hf_split is not None:
        remaining = None if pull_limit is None else max(0, pull_limit - len(out))
        if remaining is None or remaining > 0:
            hf_screens = load_rico_screens(
                path=None,
                hf_split=config.rico_hf_split,
                limit=remaining if remaining is not None else 200,
                hf_cache_path=config.rico_hf_cache_path,
            )
            _append(hf_screens, id_prefix="rico_hf")

    return out


def build_test_data(config: TestDataConfig) -> dict:
    if config.require_train_manifest and config.train_manifest is None:
        raise ValueError(
            "train_manifest is required to guarantee test data excludes training data"
        )

    train_fps = load_train_fingerprints(config.train_manifest)
    source = (config.source or "both").lower()

    seeds: list[ExampleRecord] = []
    if source in {"fixture", "fixtures", "both"}:
        seeds.extend(_fixture_seeds(config))
    if source in {"rico", "both"}:
        seeds.extend(_rico_seeds(config))
    if source not in {"rico", "fixture", "fixtures", "both"}:
        raise ValueError(f"unknown test source {config.source!r}")

    by_suite: dict[str, list[ExampleRecord]] = {s: [] for s in config.suites}
    errors: list[dict] = []
    leakage: list[dict] = []
    fixture_leaks: list[dict] = []
    seen_ids: set[str] = set()

    for seed in seeds:
        suite = str(seed.meta.get("suite") or seed.split)
        if suite not in by_suite:
            continue
        if seed.id in seen_ids:
            continue
        try:
            normalized = _normalize(seed)
        except (ParseError, ValueError) as exc:
            errors.append({"id": seed.id, "error": str(exc)})
            continue

        reasons = find_leakage(normalized, train_fps)
        if reasons:
            item = {"id": normalized.id, "reasons": reasons, "source": normalized.source}
            leakage.append(item)
            # Hand-authored fixtures must never leak on id/prompt/openui/pair.
            # design_md-only collisions (shared system text) are skipped like RICO
            # structural openui collisions — they are not content leakage.
            hard_reasons = [r for r in reasons if r != "design_md"]
            if normalized.source != "rico" and hard_reasons:
                fixture_leaks.append({**item, "reasons": hard_reasons})
            continue

        seen_ids.add(normalized.id)
        by_suite[suite].append(normalized)

        if config.target_records is not None:
            rico_kept = sum(
                len(by_suite.get(s, [])) for s in config.rico_suites if s in by_suite
            )
            if rico_kept >= config.target_records:
                break

    # Trim rico suites to the target additional count when requested.
    if config.target_records is not None:
        remaining = int(config.target_records)
        for suite in config.rico_suites:
            if suite not in by_suite:
                continue
            take = min(remaining, len(by_suite[suite]))
            by_suite[suite] = by_suite[suite][:take]
            remaining -= take

    if fixture_leaks:
        detail = ", ".join(
            f"{item['id']}[{'+'.join(item['reasons'])}]" for item in fixture_leaks[:20]
        )
        raise ValueError(
            f"test data overlaps train data ({len(fixture_leaks)} fixture leaks): {detail}"
        )

    # Fail loudly if we could not gather enough additional RICO samples.
    if config.target_records is not None:
        rico_kept = sum(len(by_suite.get(s, [])) for s in config.rico_suites)
        if rico_kept < config.target_records:
            raise ValueError(
                f"requested {config.target_records} additional RICO test samples, "
                f"but only kept {rico_kept} after validate/leakage "
                f"(errors={len(errors)}, leakage_rejected={len(leakage)}). "
                f"Increase --rico-limit / try another --rico-hf-split."
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
        "source": source,
        "seed_path": str(config.seed_path) if config.seed_path else None,
        "rico_path": str(config.rico_path) if config.rico_path else None,
        "rico_hf_split": config.rico_hf_split,
        "rico_limit": config.rico_limit,
        "rico_hf_cache_path": (
            str(config.rico_hf_cache_path) if config.rico_hf_cache_path else None
        ),
        "target_records": config.target_records,
        "suite_counts": suite_counts,
        "total_records": sum(suite_counts.values()),
        "error_count": len(errors),
        "errors": errors[:50],
        "leakage_rejected": len(leakage),
        "train_manifest": str(config.train_manifest) if config.train_manifest else None,
        "built_at": built_at,
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "version": config.version,
        "kind": "test_data",
        "source": source,
        "suites": suite_paths,
        "stats": str(stats_path.as_posix()),
        "ids": all_ids,
        "suite_counts": suite_counts,
        "train_manifest": str(config.train_manifest) if config.train_manifest else None,
        "built_at": built_at,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "manifest": manifest,
        "stats": stats,
    }
