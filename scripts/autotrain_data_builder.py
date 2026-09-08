"""Bounded data-generation side effect for the existing repair playbook."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from slm_training.data.readiness_contract import DataReadinessRequest
from slm_training.harnesses.train_data.repair import publish_repair, repair_metadata


def build_requested_successor(*, cwd: Path, request: DataReadinessRequest, **_):
    """Execute the canonical train builder with locked generation knobs.

    No guessed growth recipe for legacy actions; missing sampling/generation
    capability stays scoped. Published originals are never touched.
    """
    from slm_training.autoresearch.engine import _data_generation_flags
    from slm_training.autoresearch.schemas import DataGenerationKnobs
    from slm_training.harness_core.bounded_process import ProcessOutcome, run_bounded_process
    from slm_training.levers import INTERRUPT_AFTER_SECONDS, KILL_GRACE_SECONDS

    if request.original.kind == "eval":
        return build_screening_successor(cwd=cwd, request=request)
    if request.original.kind != "train" or request.generation_knobs is None:
        raise ValueError("capability_unavailable:locked data generation recipe missing")
    knobs = DataGenerationKnobs.model_validate(request.generation_knobs)
    if not knobs.data_only:
        raise ValueError("data repair generation must be data_only")
    base = cwd / "outputs" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="data-repair-", dir=base))
    argv = [sys.executable, "-m", "scripts.build_train_data", "--source", "programspec",
            "--version", request.successor_id, "--immutable", "--no-publish",
            "--output-root", str(attempt / "builder"), *_data_generation_flags(knobs)]
    result = run_bounded_process(
        argv, cwd=str(cwd), interrupt_after_seconds=float(INTERRUPT_AFTER_SECONDS) - 30,
        kill_grace_seconds=float(KILL_GRACE_SECONDS))
    if result.outcome != ProcessOutcome.COMPLETED or result.returncode != 0:
        raise RuntimeError(f"data builder incomplete:{result.outcome}:{result.returncode}")
    source = attempt / "builder" / request.successor_id
    # Preserve builder output and its original feedback; mutable candidate is private.
    for name in ("quality_report.json", "synthesis_feedback.json", "rejected.jsonl"):
        if not (source / name).is_file():
            raise ValueError(f"builder missing required evidence:{name}")
    for name in ("quality_report.json", "synthesis_feedback.json"):
        json.loads((source / name).read_text())
    candidate = attempt / "candidate" / request.successor_id
    shutil.copytree(source, candidate)
    manifest = json.loads((candidate / "manifest.json").read_text())
    manifest["immutable"] = False
    (candidate / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return publish_repair(request, root=cwd, staged=candidate)


def _screening_policy(cwd, request):
    from slm_training.data.readiness_contract import evidence_digest, file_digest, scoped_path
    from slm_training.data.readiness_lineage import snapshot_paths, verify_ancestry

    policy = request.generation_knobs
    expected = {"source", "corpus", "corpus_sha256", "seed", "splits", "suite"}
    if (not isinstance(policy, dict) or set(policy) != expected
            or policy["source"] != "certified" or policy["suite"] != "smoke"
            or policy["splits"] != ["validation"] or type(policy["seed"]) is not int):
        raise ValueError("capability_unavailable:locked screening sampling policy missing")
    if request.purpose != "screening_volume" or not request.training_ancestors:
        raise ValueError("screening generation requires actual training exclusions")
    if evidence_digest(policy) != request.sampling_policy_digest:
        raise ValueError("screening sampling policy digest mismatch")
    corpus = scoped_path(cwd, policy["corpus"])
    if file_digest(corpus) != policy["corpus_sha256"]:
        raise ValueError("screening sampling corpus changed")
    verify_ancestry(cwd, request)
    manifests = [snapshot_paths(cwd, ref)[0] for ref in request.training_ancestors]
    return policy, corpus, manifests


def build_screening_successor(*, cwd: Path, request: DataReadinessRequest):
    """Append a locked independent sample, never edit seeds or sealed suites.

    The original rows survive byte-for-byte. The canonical sampler excludes the
    corpus training partition and the exact declared training ancestors. Admission
    subsequently checks all scored suites and learned tables through one owner.
    """
    from slm_training.data.readiness_lineage import read_snapshot, snapshot_paths
    from slm_training.data.splits import root_family_index
    from slm_training.dsl.schema import write_jsonl
    from slm_training.harnesses.test_data.certified import sample_certified_candidates

    policy, corpus, manifests = _screening_policy(cwd, request)
    original = read_snapshot(cwd, request.original)
    families = root_family_index(original)
    need = max(0, request.minimum_unique_cases - len(original),
               request.minimum_unique_families - len(set(families.values())))
    reserved = {row.id for ref in request.scored_suites for row in read_snapshot(cwd, ref)}
    sample = sample_certified_candidates(
        existing_ids=reserved | {row.id for row in original}, need=need,
        suite="smoke", splits=("validation",), seed=policy["seed"],
        corpus_path=corpus, train_bucket_manifest=None, extra_train_manifests=manifests,
    )
    base = cwd / "outputs" / "runs"
    base.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="screening-repair-", dir=base))
    candidate = attempt / request.successor_id
    candidate.mkdir()
    records = candidate / "records.jsonl"
    old_bytes = snapshot_paths(cwd, request.original)[1].read_bytes()
    records.write_bytes(old_bytes + (b"\n" if old_bytes and not old_bytes.endswith(b"\n") else b""))
    additions = candidate / "sampled.jsonl"
    write_jsonl(additions, sample.records)
    with records.open("ab") as handle:
        handle.write(additions.read_bytes())
    # Canonical suite readers consume the manifest mapping; DataStore relocates it.
    suites = _preserve_scored_suites(cwd, request, candidate)
    manifest = {**repair_metadata(request), "suites": {**suites, "smoke": str(records)}, "records": str(records),
                "public_regression": True, "sealed": False}
    (candidate / "manifest.json").write_text(json.dumps(manifest) + "\n")
    (candidate / "sampling_report.json").write_text(json.dumps(sample.report()) + "\n")
    _screening_quality(cwd, request, candidate, sample)
    return publish_repair(request, root=cwd, staged=candidate)


def _preserve_scored_suites(cwd, request, candidate):
    from slm_training.data.readiness_lineage import snapshot_paths
    from slm_training.data.readiness_contract import scoped_path

    original = json.loads(snapshot_paths(cwd, request.original)[0].read_text())
    suites = {}
    for name, value in original.get("suites", {}).items():
        if name == "smoke":
            continue
        if not name or Path(name).name != name:
            raise ValueError("unsafe scored suite name")
        path = Path(value)
        relative = path.relative_to(cwd).as_posix() if path.is_absolute() else value
        source = scoped_path(cwd, relative)
        destination = candidate / "suites" / name / "records.jsonl"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(source, destination)
        suites[name] = str(destination)
    return suites


def _screening_quality(cwd, request, candidate, sample):
    from slm_training.data.store import write_common_manifest
    from slm_training.harnesses.test_data.certified import write_certified_eval_sidecars
    from slm_training.harnesses.train_data.readiness import check_readiness, snapshot_input

    write_common_manifest(candidate, kind="eval", dataset_id=request.successor_id)
    ref = snapshot_input(cwd, candidate, exposure=request.original.exposure)
    observed = check_readiness(request, root=cwd, candidate=ref)
    manifest = json.loads((candidate / "manifest.json").read_text())
    counts = {name: sum(bool(line.strip()) for line in Path(path).read_text().splitlines())
              for name, path in manifest["suites"].items()}
    counts["smoke"] = observed["usable_unique_cases"]
    # Full successor admission, not a count inferred from the requested sample.
    # The publication owner independently repeats this check before publishing.
    write_certified_eval_sidecars(candidate, eval_version=request.successor_id, stats={
        "suite_counts": counts,
        "leakage_rejected": len(observed["leakage"]),
        "error_count": len(observed["errors"]), "errors": observed["errors"],
        "certified": {"sampling": sample.report(), "readiness": observed,
            "corpus": request.generation_knobs["corpus"],
            "samples": {"smoke": {"seed": request.generation_knobs["seed"],
                                  "splits": request.generation_knobs["splits"]}}},
    })


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--request-payload", required=True)
    args = parser.parse_args(argv)
    request = DataReadinessRequest.model_validate_json(args.request_payload)
    result = build_requested_successor(cwd=args.cwd.resolve(), request=request)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["published"] and result["evidence"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
