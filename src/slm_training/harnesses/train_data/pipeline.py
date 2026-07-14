"""Training-data build pipeline (RICO-first)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from slm_training.data.leakage import (
    fingerprint_design_md,
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
)
from slm_training.data.rico import load_rico_screens, screen_to_record
from slm_training.dsl.language_contract import contract_id as _contract_id
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.train_data.synth import PromptSynthesizer, get_synthesizer


@dataclass
class TrainDataConfig:
    seed_path: Path | None = None
    # Human thumbs-up promotions from the annotate playground.
    human_annotations_path: Path | None = Path("fixtures/annotations/human_train.jsonl")
    rico_path: Path | None = Path("fixtures/rico/semantic_train.jsonl")
    # rico | fixture | both | awwwards | rico+awwwards | all
    source: str = "all"
    output_root: Path = Path("outputs/train_data")
    version: str = "v1"
    synthesizer: str = "quality"
    require_split: str = "train"
    rico_hf_split: str | None = None
    rico_limit: int | None = None
    max_children: int = 6
    min_quality_score: float = 0.55
    require_design_md: bool = True
    # Compact layouts train more reliably on small TwoTower models.
    max_openui_chars: int | None = None
    max_components: int | None = None
    curriculum: bool = False
    namespace_augment: bool = False
    # Exclude train records whose layout tree matches hand-authored test fixtures.
    test_seed_path: Path | None = Path("fixtures/test_seeds.jsonl")
    # Exposure control: cap records per root parent (None = uncapped). One
    # parent otherwise receives up to 6+ rows (original + synth variants).
    max_records_per_parent: int | None = None
    # P1a: fuzzy MinHash + semantic cluster caps (after exact pair dedup).
    fuzzy_dedup: bool = False
    fuzzy_jaccard: float = 0.92
    semantic_cluster_cap: int | None = None

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.version


def _normalize_record(record: ExampleRecord) -> ExampleRecord:
    from slm_training.data.structure import strip_style_literals

    scrubbed = strip_style_literals(record.openui)
    program = validate(scrubbed)
    placeholders = list(program.placeholders) or extract_placeholders(scrubbed)
    openui = strip_style_literals(program.serialized or scrubbed.strip())
    out = ExampleRecord(
        id=record.id,
        prompt=record.prompt.strip(),
        openui=openui,
        placeholders=placeholders,
        split=record.split,
        source=record.source,
        meta={
            **dict(record.meta),
            "parser": "openuidev/lang-core",
            "structure_only": True,
            "contract_id": _contract_id(),
        },
        design_md=record.design_md,
    )
    try:
        from slm_training.dsl.design_md import attach_default_design_md

        out = attach_default_design_md(out)
    except Exception:  # noqa: BLE001
        pass
    return out


def _records_from_seed_file(
    path: Path | None,
    *,
    require_split: str,
) -> tuple[list[ExampleRecord], list[dict]]:
    if path is None or not Path(path).exists():
        return [], []
    seeds = load_jsonl(path)
    errors: list[dict] = []
    out: list[ExampleRecord] = []
    for seed in seeds:
        if seed.split != require_split:
            errors.append(
                {
                    "id": seed.id,
                    "error": f"expected split {require_split!r}, got {seed.split!r}",
                }
            )
            continue
        out.append(seed)
    return out, errors


def _records_from_fixtures(
    config: TrainDataConfig,
) -> tuple[list[ExampleRecord], list[dict]]:
    fixture_records, fixture_errors = _records_from_seed_file(
        config.seed_path, require_split=config.require_split
    )
    human_records, human_errors = _records_from_seed_file(
        config.human_annotations_path, require_split=config.require_split
    )
    return fixture_records + human_records, fixture_errors + human_errors


def _records_from_rico(
    config: TrainDataConfig,
) -> tuple[list[ExampleRecord], list[dict]]:
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


def _records_from_awwwards(
    config: TrainDataConfig,
) -> tuple[list[ExampleRecord], list[dict]]:
    from slm_training.data.awwwards import AwwwardsConfig, build_awwwards_records

    try:
        records = build_awwwards_records(
            AwwwardsConfig(
                fixture_path=Path("fixtures/awwwards/sites.jsonl"),
                max_sites=50,
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

    from slm_training.harnesses.train_data.catalog import (
        LineageIndex,
        lineage_entry,
    )

    synth = synthesizer or get_synthesizer(config.synthesizer)
    # Stable seed order before expansion.
    seeds.sort(key=lambda r: r.id)
    collected: list[ExampleRecord] = []
    # Lineage over *all* candidates so parent chains survive later filtering.
    lineage_index: LineageIndex = {}
    for seed in seeds:
        candidates = [seed, *synth.expand(seed)]
        if config.namespace_augment:
            from slm_training.harnesses.train_data.synth import NamespaceAugmentSynthesizer

            ns = NamespaceAugmentSynthesizer()
            extra: list[ExampleRecord] = []
            for candidate in list(candidates):
                extra.extend(ns.expand(candidate))
            candidates.extend(extra)
        for candidate in candidates:
            lineage_index[candidate.id] = lineage_entry(candidate)
            try:
                collected.append(_normalize_record(candidate))
            except (ParseError, ValueError) as exc:
                errors.append({"id": candidate.id, "error": str(exc)})

    from slm_training.data.quality import filter_quality

    quality_kept, quality_rejected = filter_quality(
        collected,
        min_score=config.min_quality_score,
        require_design_md=config.require_design_md,
        max_openui_chars=config.max_openui_chars,
        max_components=config.max_components,
    )

    from slm_training.data.leakage import load_reserved_test_structure_fingerprints

    reserved_test_structures = load_reserved_test_structure_fingerprints(
        config.test_seed_path
    )
    structure_reserved_rejected: list[dict] = []

    deduped: list[ExampleRecord] = []
    seen_pairs: set[str] = set()
    def _accept_record(record: ExampleRecord) -> bool:
        pair = fingerprint_pair(record.prompt, record.openui)
        if pair in seen_pairs:
            return False
        seen_pairs.add(pair)
        deduped.append(record)
        return True

    for record in quality_kept:
        structure_fp = fingerprint_openui_structure(record.openui)
        if structure_fp in reserved_test_structures:
            structure_reserved_rejected.append(
                {
                    "id": record.id,
                    "source": record.source,
                    "reason": "test_fixture_structure",
                }
            )
            continue
        _accept_record(record)

    # Final stable order.
    deduped.sort(key=lambda r: r.id)
    if config.curriculum:
        from slm_training.harnesses.quality import apply_curriculum_tags

        deduped = apply_curriculum_tags(deduped)
        from slm_training.harnesses.quality import synthesize_stress_adversarial_records

        for stress in synthesize_stress_adversarial_records():
            lineage_index[stress.id] = lineage_entry(stress)
            try:
                normalized = _normalize_record(stress)
            except (ParseError, ValueError) as exc:
                errors.append({"id": stress.id, "error": str(exc)})
                continue
            structure_fp = fingerprint_openui_structure(normalized.openui)
            if structure_fp in reserved_test_structures:
                structure_reserved_rejected.append(
                    {
                        "id": normalized.id,
                        "source": normalized.source,
                        "reason": "test_fixture_structure",
                    }
                )
                continue
            _accept_record(normalized)
        deduped.sort(key=lambda r: r.id)

    # Source-family lineage + exposure control (annotate before the cap so
    # capping can group variants under their root parent).
    from slm_training.harnesses.train_data.catalog import (
        annotate_lineage,
        apply_parent_cap,
        family_stats,
    )

    deduped = annotate_lineage(deduped, lineage_index)

    fuzzy_dropped: list[dict] = []
    semantic_dropped: list[dict] = []
    if config.fuzzy_dedup:
        from slm_training.data.dedup import apply_fuzzy_dedup

        deduped, fuzzy_dropped = apply_fuzzy_dedup(
            deduped, threshold=float(config.fuzzy_jaccard)
        )
    if config.semantic_cluster_cap:
        from slm_training.data.dedup import apply_semantic_cluster_cap

        deduped, semantic_dropped = apply_semantic_cluster_cap(
            deduped, max_per_cluster=int(config.semantic_cluster_cap)
        )

    deduped, parent_cap_dropped = apply_parent_cap(
        deduped, config.max_records_per_parent
    )
    deduped.sort(key=lambda r: r.id)
    # Split-before-derive: every record inherits its root parent's split group so
    # paraphrases / augments / edits of one program never straddle a split.
    from slm_training.data.progspec.schema import assign_split_groups

    assign_split_groups(deduped)
    source_families = family_stats(deduped)
    from slm_training.data.dedup import cluster_exposure_stats

    source_families["cluster_exposure"] = cluster_exposure_stats(deduped)

    # Fingerprint final records after every train-only transformation so the
    # leakage manifest describes the exact bytes written to records.jsonl.
    prompt_fps = {fingerprint_prompt(r.prompt) for r in deduped}
    openui_fps = {fingerprint_openui(r.openui) for r in deduped}
    structure_fps = {fingerprint_openui_structure(r.openui) for r in deduped}
    seen_pairs = {fingerprint_pair(r.prompt, r.openui) for r in deduped}
    design_md_fps = {
        fp for r in deduped if (fp := fingerprint_design_md(r.design_md)) is not None
    }

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    write_jsonl(records_path, deduped)

    quality_scores = [
        float((r.meta or {}).get("quality", {}).get("score") or 0.0) for r in deduped
    ]
    stats = {
        "version": config.version,
        "contract_id": _contract_id(),
        "source": source,
        "seed_path": str(config.seed_path) if config.seed_path else None,
        "rico_path": str(config.rico_path) if config.rico_path else None,
        "rico_hf_split": config.rico_hf_split,
        "seed_count": len(seeds),
        "collected_count": len(collected),
        "quality_rejected": len(quality_rejected),
        "quality_rejected_samples": quality_rejected[:20],
        "record_count": len(deduped),
        "error_count": len(errors),
        "errors": errors[:50],
        "synthesizer": config.synthesizer,
        "min_quality_score": config.min_quality_score,
        "max_openui_chars": config.max_openui_chars,
        "max_components": config.max_components,
        "curriculum": bool(config.curriculum),
        "structure_reserved_rejected": len(structure_reserved_rejected),
        "structure_reserved_rejected_samples": structure_reserved_rejected[:20],
        "max_records_per_parent": config.max_records_per_parent,
        "parent_cap_dropped": len(parent_cap_dropped),
        "parent_cap_dropped_samples": parent_cap_dropped[:20],
        "fuzzy_dedup": bool(config.fuzzy_dedup),
        "fuzzy_dropped": len(fuzzy_dropped),
        "fuzzy_dropped_samples": fuzzy_dropped[:20],
        "semantic_cluster_cap": config.semantic_cluster_cap,
        "semantic_dropped": len(semantic_dropped),
        "semantic_dropped_samples": semantic_dropped[:20],
        "mean_quality_score": (
            round(sum(quality_scores) / len(quality_scores), 4)
            if quality_scores
            else None
        ),
        "placeholder_vocab_size": len({p for r in deduped for p in r.placeholders}),
        "with_design_md": sum(1 for r in deduped if r.design_md),
        "component_histogram": _component_histogram(deduped),
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
        "structure_fingerprints": sorted(structure_fps),
        "pair_fingerprints": sorted(seen_pairs),
        "design_md_fingerprints": sorted(design_md_fps),
        "content_fingerprint": _content_fingerprint(deduped),
        "contract_id": _contract_id(),
        "source_families": source_families,
        "built_at": stats["built_at"],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "manifest": manifest,
        "stats": stats,
    }


def _component_histogram(records: list[ExampleRecord]) -> dict[str, int]:
    from slm_training.data.quality import component_counts

    hist: dict[str, int] = {}
    for record in records:
        for name, count in component_counts(record.openui).items():
            hist[name] = hist.get(name, 0) + count
    return dict(sorted(hist.items()))


def _content_fingerprint(records: list[ExampleRecord]) -> str:
    """Stable hash of record ids + openui + prompt (ignores built_at)."""
    import hashlib

    h = hashlib.sha256()
    for record in records:
        payload = f"{record.id}\n{record.prompt}\n{record.openui}\n"
        h.update(payload.encode("utf-8"))
    return h.hexdigest()
