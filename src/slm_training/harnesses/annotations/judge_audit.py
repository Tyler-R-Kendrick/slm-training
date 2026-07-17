"""Frozen, blinded pair-study packages kept separate from training feedback."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from slm_training.lineage.records import content_sha

_OPAQUE_ANNOTATOR = re.compile(r"ann_[a-z0-9]{8,64}\Z")
_WINNERS = {"left", "right", "tie"}
_ROLES = {"rater", "adjudicator"}
REQUIRED_STRATA = {
    "empty_vs_populated",
    "v1_v2_disagreement",
    "deterministic_pass_agentv_fail",
    "binding_placeholder_failure",
    "schema_role_failure",
    "structurally_similar",
    "multiple_valid_modes",
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _study_html(public: list[dict[str, Any]], audit_id: str) -> str:
    embedded = json.dumps(public, sort_keys=True).replace("</", "<\\/")
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{audit_id} blinded pair study</title>
<style>
body {{ font: 16px system-ui; max-width: 1100px; margin: 2rem auto; }}
.pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }}
pre {{ white-space: pre-wrap; border: 1px solid #aaa; padding: 1rem; min-height: 12rem; }}
button, input {{ font: inherit; margin: .4rem; }} .status {{ color: #555; }}
</style>
<h1>Blinded OpenUI pair study</h1>
<p>Enter only the assigned opaque annotator ID. Do not include names, email, or model guesses.</p>
<label>Annotator <input id="annotator" pattern="ann_[a-z0-9]{{8,64}}"></label>
<label>Role <select id="role"><option>rater</option><option>adjudicator</option></select></label>
<div class="status" id="status"></div>
<h2 id="prompt"></h2>
<div class="pair"><section><h3>Left</h3><pre id="left"></pre></section>
<section><h3>Right</h3><pre id="right"></pre></section></div>
<p>Winner:
<button onclick="record('left')">Left</button>
<button onclick="record('right')">Right</button>
<button onclick="record('tie')">Tie</button></p>
<label><input id="acceptableLeft" type="checkbox"> Left independently acceptable</label>
<label><input id="acceptableRight" type="checkbox"> Right independently acceptable</label>
<label>Confidence (0-1) <input id="confidence" type="number" min="0" max="1" step=".05" value=".8"></label>
<label>Reason codes (comma-separated) <input id="reasons"></label>
<p><button onclick="download()">Download JSONL labels</button></p>
<script>
const auditId = {json.dumps(audit_id)};
const pairs = {embedded};
let index = 0, started = Date.now(), labels = [];
function show() {{
  const pair = pairs[index];
  status.textContent = `${{index + 1}} / ${{pairs.length}} (${{pair.pair_id}})`;
  prompt.textContent = pair.prompt; left.textContent = pair.left_openui;
  right.textContent = pair.right_openui; started = Date.now();
}}
function record(winner) {{
  const annotatorId = annotator.value;
  if (!/^ann_[a-z0-9]{{8,64}}$/.test(annotatorId)) return alert("Use the assigned opaque annotator ID.");
  labels.push({{schema:"BlindJudgeLabelV1", audit_id:auditId,
    pair_id:pairs[index].pair_id, annotator_id:annotatorId, role:role.value, winner,
    acceptable_left:acceptableLeft.checked, acceptable_right:acceptableRight.checked,
    reasons:reasons.value.split(",").map(x=>x.trim()).filter(Boolean),
    confidence:Number(confidence.value), duration_ms:Date.now()-started,
    submitted_at:new Date().toISOString()}});
  acceptableLeft.checked = false; acceptableRight.checked = false; reasons.value = "";
  if (index + 1 < pairs.length) {{ index++; show(); }} else status.textContent = "Complete. Download labels.";
}}
function download() {{
  const blob = new Blob([labels.map(x=>JSON.stringify(x)).join("\\n")+"\\n"], {{type:"application/jsonl"}});
  const link = document.createElement("a"); link.href=URL.createObjectURL(blob);
  link.download=`${{auditId}}-${{annotator.value}}.jsonl`; link.click(); URL.revokeObjectURL(link.href);
}}
show();
</script>
"""


def freeze_blinded_pairs(
    rows: list[dict[str, Any]],
    *,
    audit_id: str,
    seed: int,
    output_dir: Path,
    private_key_path: Path,
    enforce_campaign_coverage: bool = True,
    training_record_ids: set[str] | None = None,
    training_records_sha256: str | None = None,
) -> dict[str, Any]:
    """Freeze public pairs and a separately located private unblinding key."""
    if private_key_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError("private unblinding key must be outside the redacted package")
    if len({str(row["pair_id"]) for row in rows}) != len(rows):
        raise ValueError("pair_id values must be unique")
    if enforce_campaign_coverage:
        if training_record_ids is None or training_records_sha256 is None:
            raise ValueError("campaign freeze requires a pinned training-record manifest")
        if len(training_records_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in training_records_sha256
        ):
            raise ValueError("training_records_sha256 must be a lowercase sha256")
        validate_campaign_sample(rows, training_record_ids=training_record_ids)
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["pair_id"])):
        pair_id = str(row["pair_id"])
        prompt = str(row["prompt"])
        first = dict(row["candidate_a"])
        second = dict(row["candidate_b"])
        chooser = random.Random(f"{audit_id}:{seed}:{pair_id}")
        if chooser.randrange(2):
            left, right = second, first
        else:
            left, right = first, second
        public.append(
            {
                "schema": "BlindJudgePairV1",
                "audit_id": audit_id,
                "pair_id": pair_id,
                "stratum": str(row["stratum"]),
                "prompt": prompt,
                "left_openui": str(left["openui"]),
                "right_openui": str(right["openui"]),
                "prompt_sha256": _sha(prompt),
                "left_openui_sha256": _sha(str(left["openui"])),
                "right_openui_sha256": _sha(str(right["openui"])),
                "automatic_judgments_visible": False,
                "training_use": False,
            }
        )
        private.append(
            {
                "audit_id": audit_id,
                "pair_id": pair_id,
                "source_record_id": str(row["source_record_id"]),
                "left_candidate_id": str(left["candidate_id"]),
                "right_candidate_id": str(right["candidate_id"]),
                "left_family": str(left["model_family"]),
                "right_family": str(right["model_family"]),
                "left_run_id": str(left["run_id"]),
                "right_run_id": str(right["run_id"]),
                "left_checkpoint_sha256": str(left["checkpoint_sha256"]),
                "right_checkpoint_sha256": str(right["checkpoint_sha256"]),
            }
        )
    public_path = output_dir / "blind_pairs.jsonl"
    _write_jsonl(public_path, public)
    _write_jsonl(private_key_path, private)
    (output_dir / "index.html").write_text(
        _study_html(public, audit_id), encoding="utf-8"
    )
    manifest = {
        "schema": "BlindJudgeAuditManifestV1",
        "audit_id": audit_id,
        "seed": seed,
        "n": len(public),
        "public_pairs_sha256": _sha(public_path.read_text(encoding="utf-8")),
        "private_key_sha256": _sha(private_key_path.read_text(encoding="utf-8")),
        "content_sha256": content_sha(public),
        "training_use": False,
        "identity_fields_distributed": False,
        "automatic_judgments_distributed": False,
        "audit_holdout": True,
        "strata": sorted({str(row["stratum"]) for row in public}),
        "training_records_sha256": training_records_sha256,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def validate_campaign_sample(
    rows: list[dict[str, Any]], *, training_record_ids: set[str]
) -> None:
    """Enforce the powered EFS0-04 family, stratum, and holdout contract."""
    if not 90 <= len(rows) <= 110:
        raise ValueError("campaign audit sample must contain 90-110 pairs")
    strata = {str(row.get("stratum") or "") for row in rows}
    missing = REQUIRED_STRATA - strata
    if missing:
        raise ValueError(f"campaign audit sample is missing strata: {sorted(missing)}")
    families = {
        str(candidate.get("model_family") or "").lower()
        for row in rows
        for candidate in (row.get("candidate_a") or {}, row.get("candidate_b") or {})
    }
    if len(families) < 5:
        raise ValueError("campaign audit sample requires at least five model families")
    for required in ("x22", "twotower", "choice"):
        if not any(required in family for family in families):
            raise ValueError(f"campaign audit sample requires a {required} family")
    for row in rows:
        if row.get("training_use") is not False or row.get("audit_holdout") is not True:
            raise ValueError("every campaign pair must be frozen as a non-training holdout")
        source_record_id = str(row.get("source_record_id") or "")
        if not source_record_id:
            raise ValueError("every campaign pair requires source_record_id")
        if source_record_id in training_record_ids:
            raise ValueError("audit source records overlap the pinned training manifest")
        for candidate_name in ("candidate_a", "candidate_b"):
            candidate = row.get(candidate_name)
            if not isinstance(candidate, dict):
                raise ValueError(f"{candidate_name} must be an object")
            checkpoint = str(candidate.get("checkpoint_sha256") or "")
            if len(checkpoint) != 64 or any(
                char not in "0123456789abcdef" for char in checkpoint
            ):
                raise ValueError(f"{candidate_name}.checkpoint_sha256 must be immutable")


def _validate_label(row: dict[str, Any], audit_id: str) -> dict[str, Any]:
    if row.get("schema") != "BlindJudgeLabelV1" or row.get("audit_id") != audit_id:
        raise ValueError("label schema or audit_id mismatch")
    annotator_id = str(row.get("annotator_id") or "")
    if not _OPAQUE_ANNOTATOR.fullmatch(annotator_id):
        raise ValueError("annotator_id must be opaque (ann_ plus 8-64 lowercase characters)")
    if row.get("role") not in _ROLES:
        raise ValueError("role must be rater or adjudicator")
    if row.get("winner") not in _WINNERS:
        raise ValueError("winner must be left, right, or tie")
    confidence = row.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("confidence must be in [0, 1]")
    if not isinstance(row.get("duration_ms"), int) or row["duration_ms"] < 0:
        raise ValueError("duration_ms must be a non-negative integer")
    if not isinstance(row.get("acceptable_left"), bool) or not isinstance(
        row.get("acceptable_right"), bool
    ):
        raise ValueError("candidate acceptability labels must be booleans")
    reasons = row.get("reasons")
    if not isinstance(reasons, list) or any(
        not isinstance(reason, str) or not reason.strip() for reason in reasons
    ):
        raise ValueError("reasons must be a list of non-empty reason codes")
    try:
        submitted_at = datetime.fromisoformat(
            str(row.get("submitted_at") or "").replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError("submitted_at must be ISO-8601") from exc
    if submitted_at.tzinfo is None:
        raise ValueError("submitted_at must include a timezone")
    allowed = {
        "schema",
        "audit_id",
        "pair_id",
        "annotator_id",
        "role",
        "winner",
        "acceptable_left",
        "acceptable_right",
        "reasons",
        "confidence",
        "duration_ms",
        "submitted_at",
    }
    extras = set(row) - allowed
    if extras:
        raise ValueError(f"label contains forbidden identity or metadata fields: {sorted(extras)}")
    return dict(row)


def import_blinded_labels(
    label_paths: list[Path],
    *,
    audit_id: str,
    pair_ids: set[str],
    output_path: Path,
) -> dict[str, Any]:
    """Validate, deduplicate, and aggregate blind labels without unblinding."""
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    annotator_roles: dict[str, str] = {}
    for path in label_paths:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = _validate_label(json.loads(line), audit_id)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            pair_id = str(row.get("pair_id") or "")
            if pair_id not in pair_ids:
                raise ValueError(f"{path}:{line_no}: unknown pair_id {pair_id!r}")
            key = (pair_id, str(row["annotator_id"]))
            if key in labels and labels[key] != row:
                raise ValueError(f"{path}:{line_no}: conflicting duplicate label {key}")
            labels[key] = row
            previous_role = annotator_roles.setdefault(
                str(row["annotator_id"]), str(row["role"])
            )
            if previous_role != row["role"]:
                raise ValueError(
                    f"{path}:{line_no}: annotator role changed within the study"
                )

    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labels.values():
        by_pair[str(row["pair_id"])].append(row)
    aggregates = []
    incomplete = []
    for pair_id in sorted(pair_ids):
        rows = by_pair[pair_id]
        raters = [row for row in rows if row["role"] == "rater"]
        adjudicators = [row for row in rows if row["role"] == "adjudicator"]
        rater_winners = {str(row["winner"]) for row in raters}
        needs_adjudication = len(rater_winners) > 1
        adjudication_valid = (needs_adjudication and len(adjudicators) == 1) or (
            not needs_adjudication and not adjudicators
        )
        complete = len({row["annotator_id"] for row in raters}) >= 2 and adjudication_valid
        consensus = (
            str(adjudicators[0]["winner"])
            if needs_adjudication and len(adjudicators) == 1
            else (next(iter(rater_winners)) if len(rater_winners) == 1 else None)
        )
        if not complete:
            incomplete.append(pair_id)
        aggregates.append(
            {
                "pair_id": pair_id,
                "rater_n": len(raters),
                "adjudicator_n": len(adjudicators),
                "needs_adjudication": needs_adjudication,
                "complete": complete,
                "consensus_winner": consensus,
                "acceptable_left_votes": sum(row["acceptable_left"] for row in raters),
                "acceptable_right_votes": sum(row["acceptable_right"] for row in raters),
                "reason_counts": dict(
                    sorted(
                        {
                            reason: sum(reason in row["reasons"] for row in raters)
                            for reason in {
                                reason for row in raters for reason in row["reasons"]
                            }
                        }.items()
                    )
                ),
                "mean_confidence": (
                    sum(float(row["confidence"]) for row in raters) / len(raters)
                    if raters
                    else None
                ),
                "mean_duration_ms": (
                    sum(int(row["duration_ms"]) for row in raters) / len(raters)
                    if raters
                    else None
                ),
            }
        )
    payload = {
        "schema": "BlindJudgeAggregateV1",
        "audit_id": audit_id,
        "pair_n": len(pair_ids),
        "label_n": len(labels),
        "complete_pair_n": sum(row["complete"] for row in aggregates),
        "incomplete_pair_ids": incomplete,
        "pairs": aggregates,
    }
    _write_json(output_path, payload)
    return payload


def freeze_matched_training_rows(
    rows: list[dict[str, Any]],
    *,
    intersection_ids: set[str],
    audit_holdout_ids: set[str],
    seeds: tuple[int, ...],
    target_token_exposure: int,
    external_evidence_sha256: str,
    human_aggregate_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Freeze original/random/intersection arms only after admission evidence exists."""
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("matched retrain requires at least three distinct seeds")
    if target_token_exposure < 1:
        raise ValueError("target_token_exposure must be positive")
    for name, digest in (
        ("external_evidence_sha256", external_evidence_sha256),
        ("human_aggregate_sha256", human_aggregate_sha256),
    ):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"{name} must be a pinned sha256")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if not record_id or record_id in by_id:
            raise ValueError("training rows require unique record_id values")
        if _contains_judge_payload(row):
            raise ValueError("judge verdicts, rationales, and scores cannot enter training rows")
        by_id[record_id] = row
    if audit_holdout_ids & set(by_id):
        raise ValueError("audit holdout records cannot enter any training arm")
    if not intersection_ids or not intersection_ids <= set(by_id):
        raise ValueError("intersection IDs must be a non-empty subset of training rows")

    original = [by_id[record_id] for record_id in sorted(by_id)]
    intersection = [by_id[record_id] for record_id in sorted(intersection_ids)]
    arm_rows: list[dict[str, Any]] = []
    for seed in seeds:
        random_ids = sorted(
            random.Random(seed).sample(sorted(by_id), len(intersection_ids))
        )
        arms = {
            "original": original,
            "random_size_matched": [by_id[record_id] for record_id in random_ids],
            "intersection": intersection,
        }
        for arm, selected in arms.items():
            path = output_dir / f"seed-{seed}" / f"{arm}.jsonl"
            _write_jsonl(path, selected)
            arm_rows.append(
                {
                    "seed": seed,
                    "arm": arm,
                    "row_n": len(selected),
                    "target_token_exposure": target_token_exposure,
                    "rows_sha256": _sha(path.read_text(encoding="utf-8")),
                    "record_ids": [str(row["record_id"]) for row in selected],
                }
            )
    manifest = {
        "schema": "MatchedIntersectionRetrainManifestV1",
        "external_evidence_sha256": external_evidence_sha256,
        "human_aggregate_sha256": human_aggregate_sha256,
        "audit_holdout_record_ids": sorted(audit_holdout_ids),
        "seeds": list(seeds),
        "target_token_exposure": target_token_exposure,
        "identical_initialization_by_seed": True,
        "judge_payload_in_training": False,
        "arms": arm_rows,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _contains_judge_payload(value: Any) -> bool:
    forbidden = {"judge", "verdict", "rationale", "reason_codes", "score"}
    if isinstance(value, dict):
        return bool(forbidden & set(value)) or any(
            _contains_judge_payload(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_judge_payload(item) for item in value)
    return False
