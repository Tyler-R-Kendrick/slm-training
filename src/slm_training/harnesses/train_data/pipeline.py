"""Training-data build pipeline (RICO-first)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from collections import Counter

from slm_training.data.leakage import (
    fingerprint_design_md,
    fingerprint_openui,
    fingerprint_openui_structure,
    fingerprint_pair,
    fingerprint_prompt,
)
from slm_training.data.rico import load_rico_screens, screen_to_record
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.parser import ParseError, validate, validate_output
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.train_data.report import (
    build_quality_report,
    rejection_entry,
    write_quality_report,
    write_rejected,
)
from slm_training.harnesses.train_data.synth import PromptSynthesizer, get_synthesizer


@dataclass
class TrainDataConfig:
    # Curation profile: "strict" (default) turns the full dedup/verification
    # stack on; "permissive" keeps every gate at its legacy opt-in default.
    # Explicitly set fields always win over the profile (see resolve_profile).
    profile: str = "strict"
    seed_path: Path | None = None
    # Human thumbs-up promotions from the annotate playground.
    human_annotations_path: Path | None = Path(
        "src/slm_training/resources/annotations/human_train.jsonl"
    )
    rico_path: Path | None = Path(
        "src/slm_training/resources/rico/semantic_train.jsonl"
    )
    # rico | fixture | existing | both | awwwards | rico+awwwards | all
    source: str = "all"
    # Reuse a previously built records.jsonl as roots for deterministic variants.
    derive_from: Path | None = None
    output_root: Path = Path("outputs/data/train")
    version: str = "v1"
    synthesizer: str = "quality"
    require_split: str = "train"
    rico_hf_split: str | None = None
    rico_limit: int | None = None
    max_children: int = 6
    min_quality_score: float = 0.55
    min_verification_tier: str | None = None
    require_design_md: bool = True
    # Compact layouts train more reliably on small TwoTower models.
    max_openui_chars: int | None = None
    max_components: int | None = None
    curriculum: bool = False
    namespace_augment: bool = False
    # Make the declared slot contract visible to production-like training
    # prompts; this is intentionally opt-in so historical snapshots remain
    # immutable and comparable.
    prompt_slot_contract: bool = False
    # Exclude train records whose layout tree matches hand-authored test fixtures.
    test_seed_path: Path | None = Path("src/slm_training/resources/test_seeds.jsonl")
    # Exposure control: cap records per root parent (None = uncapped). One
    # parent otherwise receives up to 6+ rows (original + synth variants).
    max_records_per_parent: int | None = None
    # P1a: fuzzy MinHash + semantic cluster caps (after exact pair dedup).
    fuzzy_dedup: bool = False
    fuzzy_jaccard: float = 0.92
    semantic_cluster_cap: int | None = None
    # Cross-structure semantic dedup (SemDeDup-style) + n-gram decontamination
    # against committed eval suites. None = the profile decides; an explicit
    # True/False always wins over the profile.
    semantic_dedup: bool | None = None
    # None = engine default (embeddings 0.92, lexical-tfidf fallback 0.95).
    semantic_dedup_threshold: float | None = None
    ngram_decontam: bool | None = None
    ngram_size: int = 8
    ngram_overlap_threshold: float = 0.5
    decontam_eval_root: Path | None = Path("src/slm_training/resources/data/eval")
    # Cross-corpus dedup: drop records whose exact prompt⊕openui pair already
    # exists in these committed dataset ids (or explicit dataset paths).
    dedup_against: tuple[str, ...] = ()
    # P12 producer inputs. ProgramSpecs fall back to deterministic generation
    # when the configured file has not been materialized yet.
    programspec_path: Path | None = Path("outputs/data/programspec/programs.jsonl")
    programspec_count: int = 16
    programspec_seed: int = 0
    include_language_contract: bool = True
    deconstruct_path: Path | None = Path(
        "src/slm_training/resources/deconstruct/pipeline.jsonl"
    )
    render_path: Path | None = Path(
        "src/slm_training/resources/render/sample_program.json"
    )
    frontier_artifact_root: Path | None = Path("src/slm_training/resources/frontier")
    include_frontier_artifacts: bool = True
    repairs_per_program: int = 1
    include_edit_derivatives: bool = True
    include_scope_derivatives: bool = False
    # Scope-graded families (identity anchors / canonical pairs / scoped
    # repairs / typed lexical maps) derived per AST scope from progspec roots.
    include_scope_corpus: bool = True
    scope_kinds: tuple[str, ...] = ("document", "statement", "expression", "lexical")
    scope_identity_per_scope: int = 3
    scope_canonical_pairs_per_scope: int = 3
    scoped_repairs_per_scope: int = 2
    typed_lexical_per_program: int = 4
    # Canonical-bias ranking pairs written to preference_pairs.jsonl.
    emit_preference_pairs: bool = True
    include_design_md_contrastive: bool = True
    diffusion_online: bool = True
    governance_artifacts: bool = True
    mixture_manifest: bool = True
    # Autoresearch snapshots must never overwrite prior evidence.
    immutable: bool = False

    @property
    def output_dir(self) -> Path:
        return self.output_root / self.version


# Profile-controlled knobs. A knob is only applied when the config still holds
# the dataclass default for that field, so explicit caller/CLI choices survive.
PROFILES: dict[str, dict[str, Any]] = {
    "strict": {
        "fuzzy_dedup": True,
        "semantic_cluster_cap": 8,
        "min_verification_tier": "Bronze",
        "max_records_per_parent": 6,
        "semantic_dedup": True,
        "ngram_decontam": True,
    },
    "permissive": {},
}


def resolve_profile(config: TrainDataConfig) -> TrainDataConfig:
    """Fill profile-controlled knobs that were left at their field defaults."""
    overrides = PROFILES.get(config.profile)
    if overrides is None:
        raise ValueError(
            f"unknown train-data profile {config.profile!r}; "
            f"expected one of {sorted(PROFILES)}"
        )
    defaults = {field.name: field.default for field in fields(TrainDataConfig)}
    updates = {
        name: value
        for name, value in overrides.items()
        if getattr(config, name) == defaults[name]
    }
    return replace(config, **updates) if updates else config


def _normalize_record(record: ExampleRecord) -> ExampleRecord:
    from slm_training.data.contract import normalize_example_record
    from slm_training.data.progspec import ProgramSpec, emit_record
    from slm_training.data.structure import strip_style_literals
    from slm_training.data.verify import stamp_record

    if record.target_kind == "document" and record.meta.get("preserve_verbatim"):
        return _normalize_verbatim_document(record)
    record = normalize_example_record(record)
    if record.target_kind != "document":
        primary = validate_output(
            record.openui, record.target_kind, record.target_category
        )
        for target in record.accepted_outputs:
            validate_output(target.text, target.kind, target.category)
        meta = dict(record.meta)
        meta.setdefault("task", "generation")
        meta.setdefault("determinacy", "deterministic")
        meta.setdefault("tier", "Silver")
        meta["parser"] = "openui-output-contract"
        meta["structure_only"] = True
        meta["independent_judge_passed"] = True
        meta["verification_tier"] = "Silver"
        meta["failing_gate"] = None
        meta["verification"] = {
            "tier": "Silver",
            "failing_gate": None,
            "gates": [
                {"gate": "G0", "name": "lexical", "status": "pass"},
                {"gate": "G1", "name": "output_contract", "status": "pass"},
                {
                    "gate": "G2",
                    "name": "document_schema",
                    "status": "skip",
                    "detail": f"{record.target_kind} target",
                },
            ],
        }
        surfaces = [primary, *(target.text for target in record.accepted_outputs)]
        return ExampleRecord(
            id=record.id,
            prompt=record.prompt.strip(),
            openui=primary,
            placeholders=sorted(
                {slot for surface in surfaces for slot in extract_placeholders(surface)}
            ),
            split=record.split,
            source=record.source,
            meta=meta,
            design_md=record.design_md,
            target_kind=record.target_kind,
            target_category=record.target_category,
            accepted_outputs=list(record.accepted_outputs),
        )

    scrubbed = strip_style_literals(record.openui)
    program = validate(scrubbed)
    placeholders = list(program.placeholders) or extract_placeholders(scrubbed)
    openui = strip_style_literals(program.serialized or scrubbed.strip())
    meta = dict(record.meta)
    original_meta_keys = set(meta)
    if "verification" in meta:
        meta["upstream_verification"] = meta["verification"]
    root_id = str(meta.get("parent_id") or record.id)
    meta.setdefault("program_family_id", f"{record.source}:{root_id}")
    meta.setdefault("lineage_id", root_id)
    meta.setdefault("split_group_id", root_id)
    meta.setdefault("task", "generation")
    meta.setdefault("determinacy", "deterministic")
    meta.setdefault("parent_id", root_id)
    meta.setdefault("provenance", {})
    prompt = record.prompt.strip()
    # Remediate only edit-derived generation prompts: a raw edit instruction
    # (e.g. "Update caption content.") repurposed as a full-generation prompt is
    # under-specified, so replace it with the AST semantic contract. Authored,
    # paraphrase, abstraction-ladder, frontier and language-contract prompts stay
    # verbatim so prompt diversity (and its source families) survives dedup.
    if str(meta["task"]) == "generation" and isinstance(meta.get("edit"), dict):
        from slm_training.data.quality import (
            render_semantic_contract_prompt,
            semantic_contract_for_openui,
        )

        semantic_contract = semantic_contract_for_openui(openui)
        meta["semantic_contract"] = semantic_contract
        meta["prompt_remediation"] = {
            "kind": "ast_semantic_contract_v1",
            "original_prompt_fingerprint": fingerprint_prompt(prompt),
        }
        prompt = render_semantic_contract_prompt(semantic_contract)
    spec = ProgramSpec.from_openui(
        id=root_id,
        openui=openui,
        facts=dict(meta.get("facts") or {}),
        program_family_id=str(meta["program_family_id"]),
        lineage_id=str(meta["lineage_id"]),
        split_group_id=str(meta["split_group_id"]),
        split=record.split,
        provenance=dict(meta.get("provenance") or {}),
    )
    emitted = emit_record(
        spec,
        prompt=prompt,
        task=str(meta["task"]),
        openui=openui,
        record_id=record.id,
        parent_id=root_id,
        source=record.source,
        determinacy=str(meta["determinacy"]),
        tier=str(meta.get("tier") or "Silver"),
        meta={**meta, "parser": "openuidev/lang-core", "structure_only": True},
    )
    emitted_meta = dict(emitted.meta)
    for key in (
        "program_family_id",
        "lineage_id",
        "split_group_id",
        "task",
        "determinacy",
        "provenance",
    ):
        if key not in original_meta_keys:
            emitted_meta.pop(key, None)
    if "parent_id" not in original_meta_keys:
        emitted_meta.pop("parent_id", None)
    out = ExampleRecord(
        id=emitted.id,
        prompt=emitted.prompt,
        openui=emitted.openui,
        placeholders=placeholders,
        split=emitted.split,
        source=emitted.source,
        meta=emitted_meta,
        design_md=record.design_md,
        target_kind=record.target_kind,
        target_category=record.target_category,
        accepted_outputs=list(record.accepted_outputs),
    )
    try:
        from slm_training.dsl.design_md import attach_default_design_md

        out = attach_default_design_md(out)
    except Exception:  # noqa: BLE001
        pass
    from slm_training.data.quality import independent_judge

    # Feed the deterministic prompt/output judge into the authoritative
    # verification context; otherwise G11 is recorded as "skip" and the
    # training admission gate cannot distinguish judged from unjudged rows.
    judge = independent_judge(out)
    out.meta["independent_judge_passed"] = bool(judge["ok"])
    # Re-run F2 after F1 projection even when a producer supplied an earlier stamp.
    return stamp_record(out)


def _write_scope_preference_pairs(out_dir: Path, scope_pairs: list) -> Path:
    """Project canonical-bias scope pairs to the preference-pair contract."""
    from slm_training.harnesses.preference import PreferencePair, write_pairs

    pairs = [
        PreferencePair(
            prompt=pair.prompt,
            chosen=pair.chosen,
            rejected=pair.rejected,
            chosen_score=1.0,
            rejected_score=0.5,
            meta={
                "pair_corpus": "canonical_bias",
                "rank_source": "deterministic_canonicalization",
                "scope": pair.scope,
                "root_id": pair.root_id,
                "canonical_pair_id": pair.canonical_pair_id,
                "variant_transform": pair.variant,
            },
        )
        for pair in sorted(
            scope_pairs, key=lambda item: (item.root_id, item.scope, item.prompt)
        )
    ]
    path = out_dir / "preference_pairs.jsonl"
    write_pairs(path, pairs)
    return path


def _normalize_verbatim_document(record: ExampleRecord) -> ExampleRecord:
    """Admit an identity-anchor document without mutating its target.

    The program must still parse, but the stored ``openui`` stays
    byte-identical to what the producer emitted — no style-strip, no
    re-serialization. The stamp records the skipped serialization so the
    audit trail stays honest.
    """
    from slm_training.data.verify import stamp_record

    program = validate(record.openui)
    meta = dict(record.meta)
    root_id = str(meta.get("parent_id") or record.id)
    meta.setdefault("task", "generation")
    meta.setdefault("determinacy", "deterministic")
    meta.setdefault("program_family_id", f"{record.source}:{root_id}")
    meta.setdefault("lineage_id", root_id)
    meta.setdefault("split_group_id", root_id)
    meta["parser"] = "openuidev/lang-core"
    meta["structure_only"] = True
    meta["serialization"] = "preserved_verbatim"
    out = ExampleRecord(
        id=record.id,
        prompt=record.prompt.strip(),
        openui=record.openui,
        placeholders=list(program.placeholders)
        or extract_placeholders(record.openui),
        split=record.split,
        source=record.source,
        meta=meta,
        design_md=record.design_md,
        target_kind=record.target_kind,
        target_category=record.target_category,
        accepted_outputs=list(record.accepted_outputs),
    )
    try:
        from slm_training.dsl.design_md import attach_default_design_md

        out = attach_default_design_md(out)
    except Exception:  # noqa: BLE001
        pass
    from slm_training.data.quality import independent_judge

    judge = independent_judge(out)
    out.meta["independent_judge_passed"] = bool(judge["ok"])
    return stamp_record(out)


def _with_source(record: ExampleRecord, source: str) -> ExampleRecord:
    return ExampleRecord(
        id=record.id,
        prompt=record.prompt,
        openui=record.openui,
        placeholders=list(record.placeholders),
        split=record.split,
        source=source,
        meta=dict(record.meta),
        design_md=record.design_md,
        target_kind=record.target_kind,
        target_category=record.target_category,
        accepted_outputs=list(record.accepted_outputs),
    )


def _program_edit_records(spec: Any) -> list[ExampleRecord]:
    from slm_training.data.edits import (
        EditKind,
        EditOperation,
        EditPatch,
        ProgramDocument,
        build_transition,
        emit_transition_records,
    )

    document = ProgramDocument.from_openui(spec.canonical_openui)
    for statement in document.statements:
        if statement.name == "root":
            continue
        placeholders = extract_placeholders(statement.expression)
        if not placeholders:
            continue
        placeholder = placeholders[0]
        replacement = f"{placeholder}_edited"
        patch = EditPatch(
            (
                EditOperation(
                    EditKind.REPLACE,
                    statement.name,
                    before=statement.expression,
                    after=statement.expression.replace(placeholder, replacement, 1),
                ),
            ),
            instruction=f"Update {statement.name} content.",
        )
        transition = build_transition(
            spec.canonical_openui,
            patch.instruction,
            patch,
            render_verifier=lambda source: validate(source) is not None,
        )
        return [
            _with_source(record, "edit_trajectory")
            for record in emit_transition_records(spec, transition)
        ]
    return []


def _program_repair_records(spec: Any, limit: int) -> list[ExampleRecord]:
    if limit <= 0:
        return []
    from slm_training.data.corrupt import (
        CorruptionNotApplicable,
        CorruptionOperator,
        build_corruption,
    )

    records: list[ExampleRecord] = []
    for operator in CorruptionOperator:
        try:
            case = build_corruption(spec.canonical_openui, operator)
        except (CorruptionNotApplicable, RuntimeError, ValueError):
            continue
        record = case.to_record(spec)
        meta = {
            **record.meta,
            "synth": "corruption_repair",
            "source_kind": "deterministic",
        }
        record.meta = meta
        records.append(_with_source(record, "corruption_repair"))
        if len(records) >= limit:
            break
    return records


def _load_progspecs(config: TrainDataConfig) -> tuple[list, list[dict]]:
    """Load committed ProgramSpecs or fall back to deterministic generation."""
    from slm_training.data.progspec import ProgramGenerator, ProgramSpec

    errors: list[dict] = []
    specs: list[ProgramSpec] = []
    path = config.programspec_path
    if path is not None and Path(path).exists():
        for line_number, line in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                specs.append(ProgramSpec.from_dict(json.loads(line)))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                errors.append({"id": f"programspec:{line_number}", "error": str(exc)})
    elif config.programspec_count > 0:
        result = ProgramGenerator(seed=config.programspec_seed).generate(
            config.programspec_count
        )
        specs.extend(result.programs)
        if len(specs) != config.programspec_count:
            errors.append(
                {
                    "error": "programspec generator emitted "
                    f"{len(specs)}/{config.programspec_count} requested roots"
                }
            )
    return specs, errors


def _records_from_progspec(
    config: TrainDataConfig,
    *,
    preloaded: tuple[list, list[dict]] | None = None,
) -> tuple[list[ExampleRecord], list[dict]]:
    from slm_training.data.progspec import emit_record
    from slm_training.data.verify import VerificationContext, stamp_record

    specs, load_errors = (
        preloaded if preloaded is not None else _load_progspecs(config)
    )
    errors = list(load_errors)
    out: list[ExampleRecord] = []
    for spec in sorted(specs, key=lambda item: item.id):
        if spec.split != config.require_split:
            errors.append(
                {
                    "id": spec.id,
                    "error": f"expected split {config.require_split!r}, got {spec.split!r}",
                }
            )
            continue
        try:
            record = emit_record(
                spec,
                prompt=f"Generate the {spec.program_family_id} OpenUI program.",
                task="generation",
                record_id=spec.id,
                source="programspec_generated",
                meta={"source_kind": "program-first"},
            )
            out.append(
                stamp_record(record, VerificationContext(source_kind="program-first"))
            )
            out.extend(_program_repair_records(spec, config.repairs_per_program))
            if config.include_edit_derivatives:
                out.extend(_program_edit_records(spec))
            if config.include_scope_derivatives:
                from slm_training.data.progspec import derive_scope_records

                out.extend(derive_scope_records(spec))
        except (RuntimeError, ValueError) as exc:
            errors.append({"id": spec.id, "error": str(exc)})
    return out, errors


def _records_from_language_contract(
    config: TrainDataConfig,
) -> tuple[list[ExampleRecord], list[dict]]:
    if not config.include_language_contract:
        return [], []
    from slm_training.data.language_contract import iter_positives

    return list(iter_positives(config.require_split)), []


def _records_from_scope_corpus(
    config: TrainDataConfig,
    *,
    preloaded: tuple[list, list[dict]] | None = None,
) -> tuple[list[ExampleRecord], list[dict], list]:
    """Scope-graded families (identity / canonical / repair / typed) per root."""
    if not config.include_scope_corpus:
        return [], [], []
    from slm_training.harnesses.train_data.scope_corpus import (
        ScopeCorpusConfig,
        build_scope_corpus,
    )

    specs, load_errors = (
        preloaded if preloaded is not None else _load_progspecs(config)
    )
    errors = list(load_errors)
    corpus_config = ScopeCorpusConfig(
        scopes=tuple(config.scope_kinds),
        identity_per_scope=config.scope_identity_per_scope,
        canonical_pairs_per_scope=config.scope_canonical_pairs_per_scope,
        repairs_per_scope=config.scoped_repairs_per_scope,
        typed_per_program=config.typed_lexical_per_program,
    )
    records: list[ExampleRecord] = []
    pairs: list = []
    for spec in sorted(specs, key=lambda item: item.id):
        if spec.split != config.require_split:
            continue
        try:
            spec_records, spec_pairs = build_scope_corpus(
                root_id=spec.id,
                openui=spec.canonical_openui,
                split=spec.split,
                split_group_id=spec.split_group_id,
                program_family_id=spec.program_family_id,
                lineage_id=spec.lineage_id,
                config=corpus_config,
            )
            records.extend(spec_records)
            pairs.extend(spec_pairs)
        except (ParseError, RuntimeError, ValueError) as exc:
            errors.append({"id": f"scope_corpus:{spec.id}", "error": str(exc)})
    return records, errors, pairs


def _records_from_deconstruct(
    config: TrainDataConfig,
) -> tuple[list[ExampleRecord], list[dict]]:
    """Build governed web candidates from committed, inert capture evidence."""
    path = config.deconstruct_path
    if path is None or not Path(path).exists():
        return [], []
    from slm_training.data.deconstruct import BrowserCapture, build_web_projection
    from slm_training.data.governance import SourceProvenance
    from slm_training.data.verify import RuntimeEvidence

    records: list[ExampleRecord] = []
    errors: list[dict] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if "prompt" in row and "openui" in row:
                record = ExampleRecord.from_dict(row)
                records.append(_with_source(record, "web_distilled"))
                continue
            capture_data = dict(row["capture"])
            capture = BrowserCapture(
                source_url=str(capture_data["source_url"]),
                dom_snapshot=capture_data.get("dom_snapshot"),
                accessibility_tree=tuple(capture_data.get("accessibility_tree") or ()),
                computed_layout=tuple(capture_data.get("computed_layout") or ()),
                screenshot_refs=tuple(capture_data.get("screenshot_refs") or ()),
                interaction_trace=tuple(capture_data.get("interaction_trace") or ()),
                responsive_state=str(capture_data.get("responsive_state") or "default"),
            )
            provenance_data = dict(row["provenance"])
            provenance = SourceProvenance.from_content(
                source_url=capture.source_url,
                acquisition_date=str(provenance_data["acquisition_date"]),
                terms_policy_id=str(provenance_data["terms_policy_id"]),
                legal_basis=str(provenance_data["legal_basis"]),
                license=str(provenance_data["license"]),
                attribution=str(provenance_data["attribution"]),
                asset_rights=dict(provenance_data["asset_rights"]),
                robots_policy=str(provenance_data["robots_policy"]),
                deletion_procedure=str(provenance_data["deletion_procedure"]),
                content=capture.dom_snapshot or "",
                transformation_history=tuple(
                    provenance_data.get("transformation_history") or ()
                ),
            )
            records.append(
                build_web_projection(
                    projection_id=str(row["id"]),
                    capture=capture,
                    candidate_openui=str(row["candidate_openui"]),
                    element_statuses=dict(row["element_statuses"]),
                    runtime_evidence=RuntimeEvidence.from_dict(
                        dict(row.get("runtime_evidence") or {})
                    ),
                    provenance=provenance,
                    candidate_links=dict(row.get("candidate_links") or {}),
                    matched_behaviors=int(row.get("matched_behaviors") or 0),
                    expected_behaviors=int(row.get("expected_behaviors") or 0),
                    human_reviewed=bool(row.get("human_reviewed")),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"id": f"deconstruct:{line_number}", "error": str(exc)})
    return [_with_source(record, "web_distilled") for record in records], errors


def _records_from_render(
    config: TrainDataConfig,
) -> tuple[list[ExampleRecord], list[dict]]:
    """Build a grounded visual edit from the committed renderer contract fixture."""
    path = config.render_path
    if path is None or not Path(path).exists():
        return [], []
    if Path(path).suffix == ".jsonl":
        records, errors = _records_from_seed_file(
            Path(path), require_split=config.require_split
        )
        return [_with_source(record, "renderer_visual") for record in records], errors
    from slm_training.data.edits import EditKind, EditOperation, EditPatch
    from slm_training.data.progspec import ProgramSpec
    from slm_training.data.render import (
        BoundingBox,
        CaptureVariant,
        RenderCapture,
        RenderElement,
        ScrollTile,
        VisualMarkup,
        build_visual_edit_record,
        openui_node_id,
    )
    from slm_training.dsl.language_contract import contract_id

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "prompt" in data and "openui" in data:
            return [_with_source(ExampleRecord.from_dict(data), "renderer_visual")], []
        data["contract_id"] = contract_id()
        spec = ProgramSpec.from_dict(data)

        def element(
            statement: str,
            box: BoundingBox,
            *,
            parent: str | None = "root",
            z: int = 0,
            role: str = "generic",
        ) -> RenderElement:
            return RenderElement(
                openui_node_id=openui_node_id(spec.id, statement),
                statement_name=statement,
                parent_node_id=(
                    None if parent is None else openui_node_id(spec.id, parent)
                ),
                bounding_box=box,
                visible_clip=box,
                z_order=z,
                semantic_role=role,
                accessible_name=statement,
                interaction_target=role == "button",
                render_state="populated",
            )

        capture = RenderCapture(
            program_id=spec.id,
            variant=CaptureVariant(390, 844, "light", "populated"),
            fixed_screenshot="fixture.render.fixed.png",
            full_page_screenshot="fixture.render.full.png",
            scroll_tiles=(ScrollTile("fixture.render.tile.png", 0, 0, 390, 844),),
            elements=(
                element("root", BoundingBox(0, 0, 390, 200), parent=None),
                element("title", BoundingBox(20, 20, 200, 30), z=1, role="text"),
                element("cta", BoundingBox(20, 60, 120, 40), z=2, role="button"),
            ),
            interaction_trace=("click:button",),
        )
        patch = EditPatch(
            (
                EditOperation(
                    EditKind.REPLACE,
                    "cta",
                    before='Button(":hero.cta")',
                    after='Button(":hero.secondary_cta")',
                ),
            )
        )
        record = build_visual_edit_record(
            spec,
            capture=capture,
            markup=VisualMarkup("point", ((30, 70),)),
            instruction="Change the call to action copy",
            patch=patch,
        )
        return [_with_source(record, "renderer_visual")], []
    except (KeyError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return [], [{"id": "render", "error": str(exc)}]


def _apply_governance_gate(record: ExampleRecord) -> ExampleRecord:
    base_source = record.source.split("+", 1)[0]
    if base_source not in {"awwwards", "awwwards_real", "web_distilled"}:
        return record
    governance = record.meta.get("governance")
    if (
        isinstance(governance, dict)
        and governance.get("status") == "Complete"
        and record.meta.get("provenance_complete") is True
    ):
        return record
    from slm_training.data.governance import govern_record

    return govern_record(record, None)


class _CompositeSynthesizer:
    def __init__(self, synthesizers: list[PromptSynthesizer]) -> None:
        self._synthesizers = synthesizers

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        return [row for synth in self._synthesizers for row in synth.expand(record)]


class _FrozenArtifactBuildSynthesizer:
    """Map P6 ladder subtypes onto P11's weighted abstraction-ladder family."""

    def __init__(self, root: Path | None) -> None:
        from slm_training.harnesses.train_data.synth import FrozenArtifactSynthesizer

        self._delegate = FrozenArtifactSynthesizer(root=root)

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        rows = self._delegate.expand(record)
        for row in rows:
            if "abstraction_level" not in row.meta:
                continue
            original = str(row.meta.get("synth") or "abstraction_ladder")
            row.meta = {
                **row.meta,
                "synth": "abstraction_ladder",
                "ladder_subfamily": original,
            }
            row.source = f"{record.source}+abstraction_ladder"
        return rows


class _DesignMdContrastiveSynthesizer:
    """Attach a matched DESIGN.md and identify a deterministic mismatched control."""

    def expand(self, record: ExampleRecord) -> list[ExampleRecord]:
        if record.split != "train" or (record.meta or {}).get("task") not in {
            None,
            "generation",
        }:
            return []
        from slm_training.data.design_md import extract_design_md
        from slm_training.dsl.design_md import attach_default_design_md

        matched = attach_default_design_md(record)
        negative = extract_design_md(
            title=f"Mismatched control for {record.id}", variant="a11y"
        )
        return [
            ExampleRecord(
                id=f"{record.id}_design_contrastive",
                prompt=(
                    f"{record.prompt} Follow the supplied DESIGN.md as a binding "
                    "layout contract."
                ),
                openui=record.openui,
                placeholders=list(record.placeholders),
                split=record.split,
                source=f"{record.source}+design_md_contrastive",
                meta={
                    **record.meta,
                    "synth": "design_md_contrastive",
                    "parent_id": record.id,
                    "task": "generation",
                    "design_md_contrastive": {
                        "role": "matched",
                        "negative_fingerprint": fingerprint_design_md(negative),
                    },
                },
                design_md=matched.design_md,
            )
        ]


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
                fixture_path=Path("src/slm_training/resources/awwwards/sites.jsonl"),
                max_sites=50,
            )
        )
        return records, []
    except Exception as exc:  # noqa: BLE001
        return [], [{"error": f"awwwards: {exc}"}]


def _records_from_existing(
    config: TrainDataConfig,
) -> tuple[list[ExampleRecord], list[dict]]:
    path = config.derive_from
    if path is None:
        raise ValueError("source 'existing' requires --derive-from")
    if not Path(path).is_file():
        raise ValueError(f"derivation source does not exist: {path}")
    records: list[ExampleRecord] = []
    errors: list[dict] = []
    for record in load_jsonl(path):
        family = str((record.meta or {}).get("program_family_id") or "")
        if config.include_language_contract and family.startswith("language_contract:"):
            continue
        if record.split != config.require_split:
            errors.append(
                {
                    "id": record.id,
                    "error": f"expected split {config.require_split!r}, got {record.split!r}",
                }
            )
            continue
        records.append(
            ExampleRecord(
                id=record.id,
                prompt=record.prompt,
                openui=record.openui,
                placeholders=list(record.placeholders),
                split=record.split,
                source=record.source,
                meta={
                    **record.meta,
                    "derivation_source": str(path),
                    "parent_id": (record.meta or {}).get("parent_id") or record.id,
                },
                design_md=record.design_md,
                target_kind=record.target_kind,
                target_category=record.target_category,
                accepted_outputs=list(record.accepted_outputs),
            )
        )
    return records, errors


def _existing_program_derivatives(
    record: ExampleRecord, config: TrainDataConfig
) -> list[ExampleRecord]:
    """Project edit/repair tasks from an existing corpus root."""
    if record.target_kind != "document":
        return []
    from slm_training.data.progspec import ProgramSpec

    meta = record.meta or {}
    root_id = str(meta.get("parent_id") or record.id)
    spec = ProgramSpec.from_openui(
        id=root_id,
        openui=record.openui,
        facts=dict(meta.get("facts") or {}),
        program_family_id=str(
            meta.get("program_family_id") or f"{record.source}:{root_id}"
        ),
        lineage_id=str(meta.get("lineage_id") or root_id),
        split_group_id=str(meta.get("split_group_id") or root_id),
        split=record.split,
        provenance=dict(meta.get("provenance") or {}),
    )
    out = _program_repair_records(spec, config.repairs_per_program)
    if config.include_edit_derivatives:
        out.extend(_program_edit_records(spec))
    return out


def build_train_data(
    config: TrainDataConfig,
    synthesizer: PromptSynthesizer | None = None,
) -> dict:
    """Load every enabled producer, synthesize, verify, dedupe, and write artifacts."""
    config = resolve_profile(config)
    if config.immutable and (config.output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"immutable training-data snapshot already exists: {config.output_dir}"
        )
    source = (config.source or "rico").lower()
    seeds: list[ExampleRecord] = []
    errors: list[dict] = []
    # Verifier-in-the-loop ledger: every dropped candidate lands in
    # rejected.jsonl with its stage + reason (never silently discarded).
    rejections: list[dict] = []

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
    if source == "existing":
        records, source_errors = _records_from_existing(config)
        seeds.extend(records)
        errors.extend(source_errors)
    scope_preference_pairs: list = []
    if source in {"programspec", "integrated", "all"}:
        # One committed-file parse / seeded generation feeds both consumers;
        # each still reports the load errors it reported before.
        preloaded_specs = _load_progspecs(config)
        records, source_errors = _records_from_progspec(
            config, preloaded=preloaded_specs
        )
        seeds.extend(records)
        errors.extend(source_errors)
        records, source_errors, scope_preference_pairs = _records_from_scope_corpus(
            config, preloaded=preloaded_specs
        )
        seeds.extend(records)
        errors.extend(source_errors)
    if source in {"language_contract", "integrated", "all", "existing"}:
        records, source_errors = _records_from_language_contract(config)
        seeds.extend(records)
        errors.extend(source_errors)
    if source in {"deconstruct", "integrated", "all"}:
        records, source_errors = _records_from_deconstruct(config)
        seeds.extend(records)
        errors.extend(source_errors)
    if source in {"render", "integrated", "all"}:
        records, source_errors = _records_from_render(config)
        seeds.extend(records)
        errors.extend(source_errors)
    allowed = {
        "rico",
        "fixture",
        "fixtures",
        "both",
        "awwwards",
        "existing",
        "rico+awwwards",
        "programspec",
        "language_contract",
        "deconstruct",
        "render",
        "integrated",
        "all",
    }
    if source not in allowed:
        raise ValueError(f"unknown train source {config.source!r}")
    producer_error_count = len(errors)

    from slm_training.harnesses.train_data.catalog import (
        LineageIndex,
        lineage_entry,
    )

    synths = [synthesizer or get_synthesizer(config.synthesizer)]
    if config.include_frontier_artifacts:
        synths.append(_FrozenArtifactBuildSynthesizer(config.frontier_artifact_root))
    if config.include_design_md_contrastive and source in {
        "programspec",
        "integrated",
        "all",
    }:
        synths.append(_DesignMdContrastiveSynthesizer())
    synth: PromptSynthesizer = _CompositeSynthesizer(synths)
    # Stable seed order before expansion.
    seeds.sort(key=lambda r: r.id)
    collected: list[ExampleRecord] = []
    # Lineage over *all* candidates so parent chains survive later filtering.
    lineage_index: LineageIndex = {}
    for seed in seeds:
        candidates = [seed]
        if seed.target_kind == "document" and (seed.meta or {}).get("task") in {
            None,
            "generation",
        }:
            candidates.extend(synth.expand(seed))
            if source == "existing" and (
                config.include_edit_derivatives or config.repairs_per_program > 0
            ):
                try:
                    candidates.extend(_existing_program_derivatives(seed, config))
                except (ParseError, RuntimeError, ValueError) as exc:
                    errors.append({"id": seed.id, "error": str(exc)})
                    rejections.append(
                        rejection_entry(
                            "synthesis",
                            "derivative_error",
                            record_id=seed.id,
                            detail={"error": str(exc)},
                        )
                    )
        if config.namespace_augment and seed.target_kind == "document":
            from slm_training.harnesses.train_data.synth import (
                NamespaceAugmentSynthesizer,
            )

            ns = NamespaceAugmentSynthesizer()
            extra: list[ExampleRecord] = []
            for candidate in list(candidates):
                extra.extend(ns.expand(candidate))
            candidates.extend(extra)
        for candidate in candidates:
            candidate = _apply_governance_gate(candidate)
            lineage_index[candidate.id] = lineage_entry(candidate)
            try:
                collected.append(_normalize_record(candidate))
            except (ParseError, ValueError) as exc:
                errors.append({"id": candidate.id, "error": str(exc)})
                rejections.append(
                    rejection_entry(
                        "normalize",
                        "parse_or_contract_error",
                        record=candidate,
                        detail={"error": str(exc)},
                    )
                )

    verifier_rejected: list[dict] = []
    verified: list[ExampleRecord] = []
    for record in collected:
        governance = record.meta.get("governance")
        governance_status = (
            governance.get("status") if isinstance(governance, dict) else None
        )
        if (
            record.meta.get("verification_tier") == "Quarantine"
            or governance_status == "Quarantined"
        ):
            verifier_rejected.append(
                {
                    "id": record.id,
                    "failing_gate": record.meta.get("failing_gate"),
                    "governance_status": governance_status,
                }
            )
            rejections.append(
                rejection_entry(
                    "verification",
                    "quarantine",
                    record=record,
                    detail={
                        "failing_gate": record.meta.get("failing_gate"),
                        "governance_status": governance_status,
                    },
                )
            )
            continue
        verified.append(record)

    tier_rejected: list[dict] = []
    if config.min_verification_tier:
        tier_rank = {"Bronze": 0, "Silver": 1, "Gold": 2}
        minimum = tier_rank.get(config.min_verification_tier, 1)
        tiered: list[ExampleRecord] = []
        for record in verified:
            tier = str(record.meta.get("verification_tier") or "Bronze")
            if tier_rank.get(tier, -1) < minimum:
                tier_rejected.append({"id": record.id, "verification_tier": tier})
                rejections.append(
                    rejection_entry(
                        "verification_tier",
                        "below_min_tier",
                        record=record,
                        detail={
                            "verification_tier": tier,
                            "min_verification_tier": config.min_verification_tier,
                        },
                    )
                )
            else:
                tiered.append(record)
        verified = tiered

    from slm_training.data.quality import filter_quality

    quality_kept, quality_rejected = filter_quality(
        verified,
        min_score=config.min_quality_score,
        require_design_md=config.require_design_md,
        max_openui_chars=config.max_openui_chars,
        max_components=config.max_components,
    )
    verified_by_id = {record.id: record for record in verified}
    for entry in quality_rejected:
        rejections.append(
            rejection_entry(
                "quality",
                "quality_gate_failed",
                record=verified_by_id.get(str(entry.get("id"))),
                record_id=str(entry.get("id")),
                detail={key: value for key, value in entry.items() if key != "id"},
            )
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
            rejections.append(
                rejection_entry("dedup", "exact_pair_duplicate", record_id=record.id)
            )
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
            rejections.append(
                rejection_entry(
                    "decontamination", "test_fixture_structure", record_id=record.id
                )
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
                rejections.append(
                    rejection_entry(
                        "normalize",
                        "parse_or_contract_error",
                        record=stress,
                        detail={"error": str(exc)},
                    )
                )
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
                rejections.append(
                    rejection_entry(
                        "decontamination",
                        "test_fixture_structure",
                        record_id=normalized.id,
                    )
                )
                continue
            _accept_record(normalized)

    if config.prompt_slot_contract:
        from slm_training.models.template_fill import ensure_prompt_inventory

        deduped = [
            replace(
                record,
                prompt=ensure_prompt_inventory(
                    record.prompt, list(record.placeholders or [])
                ),
            )
            for record in deduped
        ]
        deduped.sort(key=lambda r: r.id)

    # Source-family lineage + exposure control (annotate before the cap so
    # capping can group variants under their root parent).
    from slm_training.harnesses.train_data.catalog import (
        annotate_lineage,
        apply_parent_cap,
        family_stats,
    )

    deduped = annotate_lineage(deduped, lineage_index)

    def _mirror_drops(stage: str, dropped: list[dict]) -> None:
        for drop in dropped:
            rejections.append(
                rejection_entry(
                    stage,
                    str(drop.get("reason") or stage),
                    record_id=str(drop.get("id")),
                    detail={
                        key: value
                        for key, value in drop.items()
                        if key not in {"id", "reason"}
                    },
                )
            )

    cross_corpus_dropped: list[dict] = []
    if config.dedup_against:
        from slm_training.data.store import DataStore

        store = DataStore()
        index: set[str] = set()
        for value in config.dedup_against:
            base = Path(store.resolve_path("train", value))
            pairs: set[str] = set()
            manifest_path = base / "manifest.json"
            if manifest_path.is_file():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                pairs = {str(item) for item in payload.get("pair_fingerprints") or []}
            if not pairs and (base / "records.jsonl").is_file():
                pairs = {
                    fingerprint_pair(record.prompt, record.openui)
                    for record in load_jsonl(base / "records.jsonl")
                }
            if not pairs:
                raise ValueError(
                    f"--dedup-against target has no resolvable fingerprints: {value}"
                )
            index |= pairs
        remaining: list[ExampleRecord] = []
        for record in deduped:
            if fingerprint_pair(record.prompt, record.openui) in index:
                cross_corpus_dropped.append(
                    {"id": record.id, "reason": "cross_corpus_duplicate"}
                )
            else:
                remaining.append(record)
        deduped = remaining
        _mirror_drops("dedup", cross_corpus_dropped)

    decontam_flagged: list[dict] = []
    decontam_suites: list[str] = []
    if config.ngram_decontam:
        from slm_training.data.decontam import apply_ngram_decontam, load_eval_suites

        eval_suites = load_eval_suites(
            config.decontam_eval_root, test_seed_path=config.test_seed_path
        )
        decontam_suites = sorted(eval_suites)
        deduped, decontam_flagged = apply_ngram_decontam(
            deduped,
            eval_suites,
            n=int(config.ngram_size),
            overlap_threshold=float(config.ngram_overlap_threshold),
        )
        _mirror_drops("decontamination", decontam_flagged)

    fuzzy_dropped: list[dict] = []
    semantic_dropped: list[dict] = []
    if config.fuzzy_dedup:
        from slm_training.data.dedup import apply_fuzzy_dedup

        deduped, fuzzy_dropped = apply_fuzzy_dedup(
            deduped, threshold=float(config.fuzzy_jaccard)
        )
        _mirror_drops("dedup", fuzzy_dropped)
    if config.semantic_cluster_cap:
        from slm_training.data.dedup import apply_semantic_cluster_cap

        deduped, semantic_dropped = apply_semantic_cluster_cap(
            deduped, max_per_cluster=int(config.semantic_cluster_cap)
        )
        _mirror_drops("dedup", semantic_dropped)

    semantic_cosine_dropped: list[dict] = []
    semantic_engine: str | None = None
    if config.semantic_dedup:
        from slm_training.data.semantic_dedup import (
            apply_semantic_dedup,
            similarity_engine,
        )

        semantic_engine = similarity_engine()
        deduped, semantic_cosine_dropped = apply_semantic_dedup(
            deduped, threshold=config.semantic_dedup_threshold
        )
        _mirror_drops("dedup", semantic_cosine_dropped)

    deduped, parent_cap_dropped = apply_parent_cap(
        deduped,
        config.max_records_per_parent,
        # Scope-graded families multiply per-root rows by design; when they
        # are part of the build, cap within each (family, parent) group so
        # they bound exposure without evicting one another. Builds without
        # scope-corpus rows keep the original cross-family semantics.
        per_family=config.include_scope_corpus
        and source in {"programspec", "integrated", "all"},
    )
    _mirror_drops("exposure", parent_cap_dropped)
    deduped.sort(key=lambda r: r.id)
    source_families = family_stats(deduped)
    from slm_training.data.dedup import cluster_exposure_stats

    source_families["cluster_exposure"] = cluster_exposure_stats(deduped)

    from slm_training.data.selection import attach_curation_scores

    attach_curation_scores(deduped)

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

    preference_pairs_path: Path | None = None
    if config.emit_preference_pairs and scope_preference_pairs:
        preference_pairs_path = _write_scope_preference_pairs(
            out_dir, scope_preference_pairs
        )

    governance_paths: dict[str, Path] = {}
    if config.governance_artifacts:
        from slm_training.data.governance import emit_dataset_metadata

        governance_paths = emit_dataset_metadata(
            deduped,
            out_dir / "governance",
            name="OpenUI training corpus",
            version=config.version,
        )

    mixture_payload: dict | None = None
    mixture_path: Path | None = None
    if config.mixture_manifest:
        from slm_training.data.mixture import (
            DEFAULT_TASK_WEIGHTS,
            MixtureManifest,
            corpus_diagnostics,
            default_base_weights,
            mixture_hash,
        )

        mixture = MixtureManifest(
            mixture_id=f"{config.version}-base",
            weights=default_base_weights(),
            task_weights=DEFAULT_TASK_WEIGHTS,
            notes="P12 deterministic build; P11-owned weights.",
        ).normalized()
        mixture_payload = {
            "manifest": asdict(mixture),
            "hash": mixture_hash(mixture),
            "diagnostics": corpus_diagnostics(
                deduped, configured_weights=mixture.weights
            ),
        }
        mixture_path = out_dir / "mixture.json"
        mixture_path.write_text(
            json.dumps(mixture_payload, indent=2) + "\n", encoding="utf-8"
        )

    rejected_path = write_rejected(out_dir, rejections)
    synthesis_rows = _synthesis_telemetry(deduped)
    built_at = datetime.now(timezone.utc).isoformat()
    quality_report = build_quality_report(
        version=config.version,
        profile=config.profile,
        built_at=built_at,
        seed_count=len(seeds),
        collected_count=len(collected),
        admitted=deduped,
        rejections=rejections,
        source_error_count=producer_error_count,
        cluster_exposure=source_families.get("cluster_exposure") or {},
        per_family=synthesis_rows,
        engines={
            "similarity": "minhash-char4gram",
            "semantic_dedup": semantic_engine,
            "decontam": (
                f"ngram-{config.ngram_size}" if config.ngram_decontam else None
            ),
        },
        decontamination_extra=(
            {
                "ngram_flagged": len(decontam_flagged),
                "ngram_size": int(config.ngram_size),
                "ngram_overlap_threshold": float(config.ngram_overlap_threshold),
                "suites_indexed": decontam_suites,
            }
            if config.ngram_decontam
            else None
        ),
    )
    quality_report_path = write_quality_report(out_dir, quality_report)

    quality_scores = [
        float((r.meta or {}).get("quality", {}).get("score") or 0.0) for r in deduped
    ]
    stats = {
        "version": config.version,
        "profile": config.profile,
        "source": source,
        "derive_from": str(config.derive_from) if config.derive_from else None,
        "seed_path": str(config.seed_path) if config.seed_path else None,
        "rico_path": str(config.rico_path) if config.rico_path else None,
        "rico_hf_split": config.rico_hf_split,
        "seed_count": len(seeds),
        "collected_count": len(collected),
        "verifier_rejected": len(verifier_rejected),
        "verifier_rejected_samples": verifier_rejected[:20],
        "verification_tier_rejected": len(tier_rejected),
        "verification_tier_rejected_samples": tier_rejected[:20],
        "quality_rejected": len(quality_rejected),
        "quality_rejected_samples": quality_rejected[:20],
        "record_count": len(deduped),
        "error_count": len(errors),
        "errors": errors[:50],
        "rejected_total": len(rejections),
        "rejected_path": str(rejected_path.as_posix()),
        "quality_report_path": str(quality_report_path.as_posix()),
        "synthesizer": config.synthesizer,
        "min_quality_score": config.min_quality_score,
        "min_verification_tier": config.min_verification_tier,
        "max_openui_chars": config.max_openui_chars,
        "max_components": config.max_components,
        "curriculum": bool(config.curriculum),
        "prompt_slot_contract": bool(config.prompt_slot_contract),
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
        "semantic_dedup": bool(config.semantic_dedup),
        "semantic_dedup_engine": semantic_engine,
        "semantic_cosine_dropped": len(semantic_cosine_dropped),
        "semantic_cosine_dropped_samples": semantic_cosine_dropped[:20],
        "ngram_decontam": bool(config.ngram_decontam),
        "decontam_flagged": len(decontam_flagged),
        "decontam_flagged_samples": decontam_flagged[:20],
        "dedup_against": list(config.dedup_against),
        "cross_corpus_dropped": len(cross_corpus_dropped),
        "cross_corpus_dropped_samples": cross_corpus_dropped[:20],
        "producer_inputs": {
            "programspec_path": (
                str(config.programspec_path) if config.programspec_path else None
            ),
            "programspec_count": config.programspec_count,
            "programspec_seed": config.programspec_seed,
            "language_contract": bool(config.include_language_contract),
            "deconstruct_path": (
                str(config.deconstruct_path) if config.deconstruct_path else None
            ),
            "render_path": str(config.render_path) if config.render_path else None,
            "frontier_artifacts": bool(config.include_frontier_artifacts),
            "repairs_per_program": config.repairs_per_program,
            "edit_derivatives": bool(config.include_edit_derivatives),
            "scope_derivatives": bool(config.include_scope_derivatives),
            "design_md_contrastive": bool(config.include_design_md_contrastive),
            "scope_corpus": bool(config.include_scope_corpus),
            "scope_kinds": list(config.scope_kinds),
            "scope_identity_per_scope": config.scope_identity_per_scope,
            "scope_canonical_pairs_per_scope": config.scope_canonical_pairs_per_scope,
            "scoped_repairs_per_scope": config.scoped_repairs_per_scope,
            "typed_lexical_per_program": config.typed_lexical_per_program,
        },
        "preference_pairs": len(scope_preference_pairs),
        "preference_pairs_path": (
            str(preference_pairs_path) if preference_pairs_path else None
        ),
        "mixture": mixture_payload,
        "mean_quality_score": (
            round(sum(quality_scores) / len(quality_scores), 4)
            if quality_scores
            else None
        ),
        "placeholder_vocab_size": len({p for r in deduped for p in r.placeholders}),
        "with_design_md": sum(1 for r in deduped if r.design_md),
        "component_histogram": _component_histogram(deduped),
        "built_at": built_at,
    }
    stats_path = out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    synthesis_telemetry_path = out_dir / "synthesis_telemetry.jsonl"
    synthesis_telemetry_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in synthesis_rows),
        encoding="utf-8",
    )

    manifest = {
        "version": config.version,
        "kind": "train_data",
        "profile": config.profile,
        "source": source,
        "records": str(records_path.as_posix()),
        "stats": str(stats_path.as_posix()),
        "quality_report": str(quality_report_path.as_posix()),
        "rejected": str(rejected_path.as_posix()),
        "record_count": len(deduped),
        "ids": [r.id for r in deduped],
        "split_group_ids": sorted(
            {
                str(group)
                for r in deduped
                if (group := (r.meta or {}).get("split_group_id"))
            }
        ),
        "prompt_fingerprints": sorted(prompt_fps),
        "openui_fingerprints": sorted(openui_fps),
        "structure_fingerprints": sorted(structure_fps),
        "pair_fingerprints": sorted(seen_pairs),
        "design_md_fingerprints": sorted(design_md_fps),
        "content_fingerprint": _content_fingerprint(deduped),
        "source_families": source_families,
        "governance": {
            name: str(path.as_posix()) for name, path in governance_paths.items()
        },
        "mixture": str(mixture_path.as_posix()) if mixture_path else None,
        "synthesis_telemetry": str(synthesis_telemetry_path.as_posix()),
        "synthesis_telemetry_sha256": _file_sha(synthesis_telemetry_path),
        "diffusion_online": (
            asdict(_diffusion_config()) if config.diffusion_online else None
        ),
        "built_at": stats["built_at"],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "manifest": manifest,
        "stats": stats,
        "quality_report": quality_report,
        "governance": {name: str(path) for name, path in governance_paths.items()},
    }


def _diffusion_config():
    from slm_training.data.diffusion import DiffusionConfig

    return DiffusionConfig()


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


def _synthesis_telemetry(records: list[ExampleRecord]) -> list[dict[str, Any]]:
    by_family: dict[str, list[ExampleRecord]] = {}
    for record in records:
        family = str((record.meta or {}).get("source_family") or record.source)
        by_family.setdefault(family, []).append(record)
    rows = []
    for family, members in sorted(by_family.items()):
        scores = [
            float((record.meta or {}).get("quality", {}).get("score") or 0)
            for record in members
        ]
        rows.append(
            {
                "source_family": family,
                "record_count": len(members),
                "root_parent_count": len(
                    {
                        str((record.meta or {}).get("parent_id") or record.id)
                        for record in members
                    }
                ),
                "mean_quality_score": round(sum(scores) / len(scores), 6),
                "min_quality_score": min(scores),
                "max_quality_score": max(scores),
                "task_counts": dict(
                    sorted(
                        Counter(
                            str((record.meta or {}).get("task") or "generation")
                            for record in members
                        ).items()
                    )
                ),
            }
        )
    return rows


def _file_sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
