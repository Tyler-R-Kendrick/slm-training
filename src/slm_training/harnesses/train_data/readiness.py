"""Action-specific readiness using actual model-build admission and preparation.

Reports are observations, not repair acknowledgments. The controller must bind
the request/current fence and independently run this predicate before closure.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from slm_training.data.leakage import fingerprint_pair
from slm_training.data.readiness_contract import (
    DataReadinessRequest, SnapshotInput, evidence_digest, file_digest, scoped_path,
    READINESS_CONTEXT_FIELDS, READINESS_OPTIONAL_CONTEXT_FIELDS, validate_bootstrap_inputs,
)
from slm_training.data.readiness_lineage import request_leakage, snapshot_paths, snapshot_input
from slm_training.data.record_admission import assert_training_record
from slm_training.dsl.schema import ExampleRecord


def _partition_rows(path: Path, request: DataReadinessRequest):
    admitted, rejected, raw_families = [], [], Counter()
    seen_ids, seen_pairs = set(), set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        family = "unparsed"
        try:
            row = ExampleRecord.from_dict(json.loads(line))
            family = str(row.meta.get("root_parent_id") or row.id)
            raw_families[family] += 1
            assert_training_record(row)
            if row.target_kind != request.target_kind:
                raise ValueError("wrong target kind for requested activity")
            if request.original.kind == "train" and row.split != "train":
                raise ValueError("non-training row in requested training snapshot")
            pair = fingerprint_pair(row.prompt, row.openui)
            if row.id in seen_ids or pair in seen_pairs:
                raise ValueError("duplicate case id or canonical prompt/program pair")
            seen_ids.add(row.id)
            seen_pairs.add(pair)
            admitted.append(row)
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append({"line": number, "family": family,
                             "row_sha256": evidence_digest(line),
                             "reason": f"{type(exc).__name__}:{exc}"[:800]})
    return admitted, rejected, dict(raw_families)


def _trainer_preparation(path: Path, request: DataReadinessRequest) -> dict:
    """Invoke real loader and TwoTower constructor, with no training/forward.

    The request's exact configuration is used, including tokenizer options.
    This local adapter refuses any configuration that might acquire HF weights.
    """
    from slm_training.harnesses.model_build.readiness import prepare_records

    result = prepare_records(path, request.trainer_config)
    if request.additional_trainer_configs:
        result["additional_preparations"] = [prepare_records(path, cfg) for cfg in request.additional_trainer_configs]
    return result


def check_readiness(
    request: DataReadinessRequest, *, root: Path, candidate: SnapshotInput | None = None,
) -> dict:
    """Reread exact inputs; malformed/missing/incompatible evidence stays incomplete."""
    from slm_training.harnesses.model_build.readiness import ReadinessCapabilityUnavailable

    ref = candidate or request.original
    report = {"schema_version": "data_readiness_evidence/v1",
              "request_sha256": request.sha256, "snapshot": ref.model_dump(),
              "ready": False, "failure_kind": "data_failure", "errors": [], "rejected": [], "leakage": [],
              "usable_unique_cases": 0, "usable_unique_families": 0,
              "scored_suites": [r.model_dump() for r in request.scored_suites],
              "training_ancestors": [r.model_dump() for r in request.training_ancestors],
              "learned_tables": [r.model_dump() for r in request.learned_tables]}
    try:
        _check_candidate_identity(root, request, ref, candidate is not None)
        _check_ranker_sources(root, request)
        _, path = snapshot_paths(root, ref)
        rows, rejected, support = _partition_rows(path, request)
        report.update(rejected=rejected, input_family_support=support)
        from slm_training.data.splits import root_family_index

        families = root_family_index(rows)
        report.update(usable_unique_cases=len(rows),
                      usable_unique_families=len(set(families.values())),
                      admitted_family_support=dict(Counter(families.values())))
        report["leakage"] = request_leakage(root, request, rows)
        if rejected or report["leakage"]:
            report["errors"].append("invalid_or_contaminated_records")
        if len(rows) < request.minimum_unique_cases:
            report["errors"].append("insufficient_unique_cases")
        if len(set(families.values())) < request.minimum_unique_families:
            report["errors"].append("insufficient_independent_families")
        if not report["errors"]:
            report["preparation"] = _trainer_preparation(path, request)
            report["ready"] = True
            report["failure_kind"] = None
    except ReadinessCapabilityUnavailable as exc:
        report["failure_kind"] = "capability_unavailable"
        report["errors"].append(str(exc))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        report["failure_kind"] = "code_failure" if isinstance(exc, (KeyError, TypeError, RuntimeError)) else "data_failure"
        report["errors"].append(f"{type(exc).__name__}:{exc}"[:1000])
    return report


def _check_candidate_identity(root, request, ref, is_candidate):
    if not is_candidate:
        return
    snapshot_paths(root, request.original)
    if ref.dataset_id != request.successor_id or ref.kind != request.original.kind:
        raise ValueError("wrong successor dataset/role for original predicate")
    if Path(ref.directory).name != request.successor_id:
        raise ValueError("successor publication destination mismatch")
    manifest = json.loads((scoped_path(root, ref.directory) / "manifest.json").read_text())
    if manifest.get("successor_of") != request.original.manifest_sha256:
        raise ValueError("missing original snapshot lineage")
    if manifest.get("readiness_request_sha256") != request.sha256:
        raise ValueError("successor belongs to a different data action")
    if manifest.get("preprocessing_identity") != request.preprocessing_identity:
        raise ValueError("preprocessing identity changed")
    if request.purpose == "screening_volume":
        if manifest.get("sampling_policy_digest") != request.sampling_policy_digest:
            raise ValueError("screening sampling policy changed")
        old = json.loads((scoped_path(root, request.original.directory) / "manifest.json").read_text())
        if old.get("sealed") or manifest.get("sealed"):
            raise ValueError("routine screening action cannot replace sealed suite")
        _check_retained_suites(root, old, manifest)


def _check_retained_suites(root, old, current):
    for name, source in old.get("suites", {}).items():
        if name == "smoke":
            continue
        destination = current.get("suites", {}).get(name)
        if not destination:
            raise ValueError("screening successor dropped a scored suite")
        paths = [Path(value).relative_to(root).as_posix() if Path(value).is_absolute() else value
                 for value in (source, destination)]
        if file_digest(scoped_path(root, paths[0])) != file_digest(scoped_path(root, paths[1])):
            raise ValueError("screening successor changed a non-treatment suite")


def _screening_snapshots(root, train_version, eval_version):
    from slm_training.data.store import DataStore

    store = DataStore(root)
    train = store.resolve("train", train_version).path
    evaluation = store.resolve("eval", eval_version).path
    eval_manifest = json.loads((evaluation / "manifest.json").read_text(encoding="utf-8"))
    suites = eval_manifest.get("suites")
    smoke_name = suites.get("smoke") if isinstance(suites, dict) else None
    if not isinstance(smoke_name, str) or not smoke_name:
        raise ValueError("certified eval manifest has no smoke suite")
    scored = {}
    for name, value in suites.items():
        if not isinstance(value, str) or not value:
            raise ValueError("certified suite has no records input")
        # DataStore's publisher may emit absolute paths; normalize only inside
        # this root, then apply the same no-symlink/no-traversal scope check.
        relative = Path(value).relative_to(root).as_posix() if Path(value).is_absolute() else value
        path = scoped_path(root, relative)
        if not path.is_file() or not path.is_relative_to(evaluation.resolve()):
            raise ValueError("certified suite is outside its eval snapshot")
        scored[name] = snapshot_input(root, evaluation, exposure="public_regression",
                                     records=path.relative_to(evaluation.resolve()).as_posix())
    original = scored["smoke"]
    train_manifest = json.loads((train / "manifest.json").read_text(encoding="utf-8"))
    raw_records = train_manifest.get("records", "records.jsonl")
    if not isinstance(raw_records, str) or not raw_records:
        raise ValueError("train manifest has no records input")
    records = Path(raw_records)
    if not records.is_absolute():
        records = (root / records if (root / records).is_file() else train / records)
    records = records.resolve()
    if not records.is_file() or not records.is_relative_to(train.resolve()):
        raise ValueError("train records are outside their snapshot")
    ancestor = snapshot_input(
        root, train, exposure="train_only",
        records=records.relative_to(train.resolve()).as_posix(),
    )
    return evaluation, original, ancestor, list(scored.values())


def _screening_sampling(root, evaluation):
    quality = json.loads((evaluation / "quality_report.json").read_text(encoding="utf-8"))
    certified = quality.get("certified")
    sample = certified.get("samples", {}).get("smoke") if isinstance(certified, dict) else None
    corpus_name = certified.get("corpus") if isinstance(certified, dict) else None
    if not isinstance(sample, dict) or not isinstance(corpus_name, str):
        raise ValueError("certified eval metadata lacks screening sampling authority")
    corpus = (root / corpus_name).resolve()
    if not corpus.is_file():
        raise ValueError("certified screening corpus is unavailable")
    sampling = {
        "source": "certified", "corpus": corpus.relative_to(root.resolve()).as_posix(),
        "corpus_sha256": file_digest(corpus), "seed": sample.get("seed"),
        "splits": list(sample.get("splits") or ()), "suite": "smoke",
    }
    if type(sampling["seed"]) is not int or sampling["splits"] != ["validation"]:
        raise ValueError("certified screening sampling policy is incomplete")
    stamps = quality.get("version_stamp", {}).get("components", {})
    preprocessing = {key: value for key, value in stamps.items() if key.startswith("data.")}
    if not preprocessing or any(not isinstance(value, str) or not value for value in preprocessing.values()):
        raise ValueError("certified eval preprocessing identity is unavailable")
    return sampling, "data-components:" + evidence_digest(preprocessing)


def screening_request_context(context):
    """Validate controller context without inventing checkpoint or data ancestry.

    All dependency lists must be explicit, including an intentionally empty
    learned-table list. The controller owns completeness against its locked
    activity; the factory additionally checks actual input bytes and lineage.
    """
    from slm_training.models.twotower import TwoTowerConfig

    required = READINESS_CONTEXT_FIELDS
    if not isinstance(context, dict) or not required.issubset(context):
        raise ValueError("capability_unavailable:locked_data_readiness_context")
    if context.keys() - required - READINESS_OPTIONAL_CONTEXT_FIELDS:
        raise ValueError("unknown screening readiness context field")
    if not isinstance(context["trainer_config"], dict) or not context["trainer_config"]:
        raise ValueError("missing actual trainer configuration")
    normalized = {**context, "trainer_config": asdict(TwoTowerConfig(**context["trainer_config"]))}
    if "additional_trainer_configs" in context:
        additional = context["additional_trainer_configs"]
        if not isinstance(additional, list) or any(not isinstance(cfg, dict) or not cfg for cfg in additional):
            raise ValueError("additional trainer configurations must be explicit nonempty mappings")
        normalized["additional_trainer_configs"] = [asdict(TwoTowerConfig(**cfg)) for cfg in additional]
    if context["initialization"] == "scratch" and (context["lineage_root"] or context["starting_run_id"]):
        raise ValueError("scratch initialization cannot conceal a parent run")
    return normalized


def _check_ranker_sources(root, request):
    """An enabled train-derived ranker cannot hide behind an empty dependency list."""
    for config in [request.trainer_config, *request.additional_trainer_configs]:
        _check_ranker_config(root, request, config)


def _check_ranker_config(root, request, config):
    from slm_training.harnesses.model_build.readiness import configured_ranker

    if config.get("speculative_rank", "off") != "off" and not request.learned_tables:
        raise ValueError("missing configured ranker learned-table provenance")
    path = configured_ranker(config, root)
    if path is None:
        return
    table = next((ref for ref in request.learned_tables if scoped_path(root, ref.path) == path), None)
    if table is None:
        raise ValueError("missing configured ranker learned-table provenance")
    source = json.loads(path.read_text()).get("source", {})
    if not any(ref.dataset_id == source.get("dataset_id") and ref.records_sha256 == source.get("records_sha256")
               for ref in table.training_sources):
        raise ValueError("ranker training source identity mismatch")


def build_screening_request(
    *, root: Path, campaign_id: str, action_id: str, train_version: str,
    eval_version: str, minimum: int, context: dict | None = None,
) -> DataReadinessRequest:
    """Build a strict request from published inputs and certified metadata."""
    from slm_training.autoresearch.heal.fail_closed import allocate_screening_suite_id
    from slm_training.data.readiness_lineage import verify_ancestry

    if type(minimum) is not int or minimum < 1:
        raise ValueError("screening minimum must be a positive integer")
    context = screening_request_context(context)
    root = root.resolve()
    evaluation, original, ancestor, suites = _screening_snapshots(root, train_version, eval_version)
    sampling, _ = _screening_sampling(root, evaluation)
    request = DataReadinessRequest(
        action_id=action_id, campaign_id=campaign_id, purpose="screening_volume",
        original=original,
        successor_id=allocate_screening_suite_id(evaluation.parent, minimum),
        minimum_unique_cases=minimum, minimum_unique_families=minimum,
        **context,
        sampling_policy_digest=evidence_digest(sampling), generation_knobs=sampling,
    )
    if any(ref not in request.scored_suites for ref in suites) or ancestor not in request.training_ancestors:
        raise ValueError("screening context omits published scored suites or current training input")
    for ref in [*request.scored_suites, *request.training_ancestors,
                *(source for table in request.learned_tables for source in table.training_sources)]:
        snapshot_paths(root, ref)
    if any(ref.kind != "train" or ref.exposure != "train_only" for ref in request.training_ancestors):
        raise ValueError("training ancestry must contain train-only training snapshots")
    verify_ancestry(root, request)
    _check_ranker_sources(root, request)
    return request


def lock_readiness_request(store, request: DataReadinessRequest, *, experiment_id: str | None = None,
                           activity_inputs: dict | None = None):
    """Controller producer for the existing repair playbook's inline input.

    The caller resolves all suites, ancestry, minima and trainer configuration
    from its locked activity; this function must not guess them from status text.
    The campaign event is authoritative, and no mutable latest-request file is
    needed. An orphan artifact after a crash is safe to reconcile by replay.
    """
    request = DataReadinessRequest.model_validate(request.model_dump(mode="json"))
    if request.campaign_id != store.campaign_id:
        raise ValueError("wrong campaign data readiness request")
    if experiment_id is None:
        if not activity_inputs or evidence_digest(activity_inputs) != request.action_id:
            raise ValueError("bootstrap readiness requires exact controller activity inputs")
        validate_bootstrap_inputs(request, activity_inputs)
        activity = store.write_artifact("data_readiness_inputs", activity_inputs)
        binding = {"activity_input_sha256": activity.stem}
    else:
        locked = store.load_experiment_campaign(experiment_id)
        binding = {"manifest_sha256": locked.manifest_sha256}
    artifact = store.write_artifact("data_readiness_requests", request)
    event = store.append_event(
        "data_readiness_request_locked", experiment_id=experiment_id,
        artifact_sha256=artifact.stem,
        detail={"request_sha256": request.sha256, **binding},
        idempotency_key=f"data-readiness:{experiment_id}:{request.action_id}",
    )
    return {"data_readiness_request": request.model_dump(mode="json"),
            "data_readiness_request_sha256": request.sha256,
            "data_readiness_request_event_id": event["event_id"]}


def bootstrap_screening_request(*, cwd, root, loop_id, train_version, eval_version,
                               minimum, context):
    """Lock a readiness activity before any scientific experiment exists."""
    from slm_training.autoresearch.storage import CampaignStore
    from slm_training.data.readiness_contract import load_locked_readiness_request

    context = screening_request_context(context)
    # JSON normalization accepts typed snapshot references without weaker readers.
    from pydantic_core import to_jsonable_python
    inputs = to_jsonable_python({"loop_id": loop_id, "train_version": train_version,
              "eval_version": eval_version, "minimum": minimum, "context": context})
    identity = evidence_digest(inputs)
    campaign_id = "readiness-" + identity[:24]
    store = CampaignStore(campaign_id, root)
    key = f"data-readiness:None:{identity}"
    if any(event.get("idempotency_key") == key for event in store.verify_event_chain()):
        request = load_locked_readiness_request(store, action_id=identity)
    else:
        request = build_screening_request(root=Path(cwd), campaign_id=campaign_id,
            action_id=identity, train_version=train_version, eval_version=eval_version,
            minimum=minimum, context=context)
    locked = lock_readiness_request(store, request, activity_inputs=inputs)
    return {"kind": "rebuild_data", "blocker_code": "screening_suite_volume",
            "campaign_id": campaign_id, "data_action_id": identity, **locked}
