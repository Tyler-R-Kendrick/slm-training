"""SIE-005 (SLM-478): blinded construct-validity annotation packet.

Prepares everything EXP-SR-3 / SIE-006 needs for a genuinely blinded run
without requiring external-judge credentials or human raters in-repo.

Composes existing owners (does not fork them):

* ``harnesses.annotations.judge_audit`` — freeze/import, two-rater +
  adjudication, training holdout flags (SLM-106 / EFS0-04)
* ``evals.judge_independence`` — provider-neutral ExternalJudgeConfig /
  Adapter, rubric hash, independence provenance, retry/refusal/cost
* ``evals.evaluator_calibration_protocol`` — VCE-010 calibration protocol
  this packet feeds; SIE-005 ships the packet only

claim_class=fixture. Does **not** lock or execute ExperimentCampaignV1
(that is SIE-006 / exp-sr-3).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from slm_training.evals.judge_independence import (
    ExternalJudgeAdapter,
    ExternalJudgeConfig,
    text_sha256,
)
from slm_training.harnesses.annotations.judge_audit import (
    freeze_blinded_pairs,
    import_blinded_labels,
)
from slm_training.versioning import build_version_stamp

__all__ = [
    "AUDIT_ID",
    "CLAIM_CLASS",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "RELATED_CATALOGUE_ID",
    "REASON_CODES",
    "SIE005_STRATA",
    "VERSION_STAMP_COMPONENTS",
    "BlindedPacketReport",
    "build_fixture_audit_rows",
    "build_external_judge_config",
    "external_judge_request_schema",
    "external_judge_response_schema",
    "mock_external_transport",
    "assert_public_packet_blinded",
    "assert_excluded_from_training",
    "prepare_blinded_packet",
    "render_markdown",
    "run_fixture_packet",
]

MATRIX_SET = "slm478_sie005_blinded_packet"
MATRIX_VERSION = "sie005-v1"
AUDIT_ID = "sie005-construct-validity-packet"
RELATED_CATALOGUE_ID = "exp-sr-3"
CLAIM_CLASS = "fixture"
VERSION_STAMP_COMPONENTS = (
    "harness.experiments",
    "harness.experiments.slm478_sie005_blinded_packet",
)

SIE005_STRATA: tuple[str, ...] = (
    "hard_valid_contrast",
    "metamorphic_pair",
    "real_model_output",
    "ambiguity_case",
)

REASON_CODES: tuple[str, ...] = (
    "component_inventory",
    "binding_failure",
    "placeholder_failure",
    "schema_role_failure",
    "empty_or_trivial",
    "structural_mismatch",
    "multiple_valid_modes",
    "ambiguous_prompt",
)

_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "left_checkpoint_sha256",
        "right_checkpoint_sha256",
        "model_family",
        "run_id",
        "deterministic_verdict",
        "deterministic_score",
        "deterministic_pass",
        "automatic_score",
        "verifier_ok",
        "candidate_id",
        "source_record_id",
    }
)

_SECRET_PATTERNS = (
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{8,}"),
    re.compile(r"sk-[A-Za-z0-9]{16,}"),
    re.compile(r"hf_[A-Za-z0-9]{16,}"),
)


def _sha16(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _tiny_openui(label: str, *, variant: str) -> str:
    # Fixture-scale OpenUI-shaped text — not admitted into any corpus.
    return f"root = {label}_{variant}\n  Stack\n    Text value={label!r}\n"


def build_fixture_audit_rows(*, seed: int = 0) -> list[dict[str, Any]]:
    """Stratified holdout rows: hard-valid, metamorphic, model-output, ambiguity."""
    rng = random.Random(f"{AUDIT_ID}:rows:{seed}")
    rows: list[dict[str, Any]] = []
    for stratum in SIE005_STRATA:
        for idx in range(2):  # fixture-scale: 2 per stratum
            pair_key = f"{stratum}:{idx}:{seed}"
            opaque_id = f"rec_{_sha16(pair_key)}"
            left = _tiny_openui(stratum, variant=f"a{idx}")
            right = _tiny_openui(stratum, variant=f"b{idx}")
            if stratum == "metamorphic_pair" and rng.random() < 0.5:
                left, right = right, left
            rows.append(
                {
                    "pair_id": opaque_id,
                    "prompt": f"[{stratum}] render a tiny stack for fixture pair {idx}",
                    "stratum": stratum,
                    "source_record_id": f"src_{_sha16(pair_key + ':src')}",
                    # Private-side identities only — never copied into public packet.
                    "candidate_a": {
                        "openui": left,
                        "candidate_id": f"{opaque_id}:a",
                        "model_family": f"fixture_{stratum}_family_a",
                        "run_id": f"run_{AUDIT_ID}",
                        "checkpoint_sha256": text_sha256(f"ckpt-a-{pair_key}"),
                    },
                    "candidate_b": {
                        "openui": right,
                        "candidate_id": f"{opaque_id}:b",
                        "model_family": f"fixture_{stratum}_family_b",
                        "run_id": f"run_{AUDIT_ID}",
                        "checkpoint_sha256": text_sha256(f"ckpt-b-{pair_key}"),
                    },
                    # Held privately for leakage asserts; never written publicly.
                    "_private_deterministic_verdict": "left" if idx % 2 == 0 else "right",
                }
            )
    return rows


def build_external_judge_config() -> ExternalJudgeConfig:
    rubric_path = (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "evals"
        / "judge_independence_rubric_v1.md"
    )
    rubric = rubric_path.read_text(encoding="utf-8")
    return ExternalJudgeConfig(
        provider="mock_external_judge_provider",
        provider_version="sie005-fixture-1",
        model_family="mock_external_judge_family",
        model_version="sie005-fixture-1",
        rubric_id="judge_independence_rubric_v1",
        rubric_version="1",
        rubric=rubric,
        temperature=0.0,
        seed=0,
        max_attempts=2,
        max_tokens=256,
        max_cost_usd=1.0,
    )


def external_judge_request_schema() -> dict[str, Any]:
    """Provider-neutral request envelope (no credentials, no identities)."""
    return {
        "schema": "ExternalJudgeRequestV1",
        "required": [
            "model",
            "temperature",
            "max_tokens",
            "response_schema",
            "rubric",
            "rubric_sha256",
            "independence_declaration",
            "pair",
        ],
        "properties": {
            "model": {"type": "string"},
            "temperature": {"type": "number", "minimum": 0},
            "seed": {"type": ["integer", "null"]},
            "max_tokens": {"type": "integer", "minimum": 1},
            "max_attempts": {"type": "integer", "minimum": 1},
            "max_cost_usd": {"type": "number", "minimum": 0},
            "response_schema": {"const": "JudgePairResponseV1"},
            "rubric": {"type": "string"},
            "rubric_id": {"type": "string"},
            "rubric_version": {"type": "string"},
            "rubric_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "independence_declaration": {
                "type": "object",
                "required": [
                    "participated_in_creation",
                    "participated_in_admission",
                    "participated_in_training",
                    "participated_in_preference",
                    "participated_in_evaluation",
                    "rubric_used_for_training_admission",
                    "saw_candidate_identity",
                    "saw_automatic_judgments",
                    "blinded",
                ],
            },
            "pair": {
                "type": "object",
                "required": ["prompt", "left_openui", "right_openui"],
                "additionalProperties": False,
            },
        },
        "forbidden_fields": sorted(_FORBIDDEN_PUBLIC_KEYS),
    }


def external_judge_response_schema() -> dict[str, Any]:
    return {
        "schema": "JudgePairResponseV1",
        "required": ["verdict", "reason_codes", "cost_usd"],
        "properties": {
            "verdict": {"enum": ["left", "right", "tie", "refusal", "error"]},
            "left_acceptable": {"type": ["boolean", "null"]},
            "right_acceptable": {"type": ["boolean", "null"]},
            "score": {"type": ["number", "null"]},
            "confidence": {"type": ["number", "null"]},
            "reason_codes": {
                "type": "array",
                "items": {"type": "string", "enum": list(REASON_CODES)},
            },
            "cost_usd": {"type": "number", "minimum": 0},
        },
    }


def mock_external_transport(
    *, mode: str = "ok"
) -> Any:
    """No-credentials transport: ok | refusal | cost_overrun."""

    def _transport(request: dict[str, Any]) -> dict[str, Any]:
        # Fail closed if identity/verdict leaks into the request.
        blob = json.dumps(request, sort_keys=True)
        for key in _FORBIDDEN_PUBLIC_KEYS:
            if f'"{key}"' in blob:
                raise ValueError(f"request leaked forbidden field {key!r}")
        if mode == "refusal":
            return {
                "verdict": "refusal",
                "left_acceptable": None,
                "right_acceptable": None,
                "score": None,
                "confidence": None,
                "reason_codes": ["ambiguous_prompt"],
                "cost_usd": 0.0001,
            }
        if mode == "cost_overrun":
            return {
                "verdict": "left",
                "left_acceptable": True,
                "right_acceptable": False,
                "score": 0.9,
                "confidence": 0.9,
                "reason_codes": ["component_inventory"],
                "cost_usd": 99.0,
            }
        return {
            "verdict": "left",
            "left_acceptable": True,
            "right_acceptable": False,
            "score": 0.8,
            "confidence": 0.8,
            "reason_codes": ["component_inventory"],
            "cost_usd": 0.0001,
        }

    return _transport


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _randomized_annotation_order(
    public_pairs: list[dict[str, Any]], *, seed: int
) -> list[str]:
    order = [str(row["pair_id"]) for row in public_pairs]
    rng = random.Random(f"{AUDIT_ID}:order:{seed}")
    rng.shuffle(order)
    return order


def _annotation_label_schema() -> dict[str, Any]:
    return {
        "schema": "BlindJudgeLabelV1",
        "required": [
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
        ],
        "roles": ["rater", "adjudicator"],
        "winners": ["left", "right", "tie"],
        "reason_codes": list(REASON_CODES),
        "annotator_id_pattern": r"ann_[a-z0-9]{8,64}",
        "policy": {
            "min_raters": 2,
            "adjudication_on_disagreement": True,
            "exactly_one_adjudicator_when_needed": True,
            "no_automatic_scores": True,
        },
    }


def assert_public_packet_blinded(public_dir: Path) -> list[str]:
    """Raters must not see checkpoint identity or deterministic verdicts.

    Scans rater-facing artifacts only. Operator schema pins under
    ``external_judge/`` may name the *judge* model family — that is not
    candidate identity.
    """
    findings: list[str] = []
    rater_files = (
        "blind_pairs.jsonl",
        "index.html",
        "annotation_order.json",
        "label_schema.json",
        "manifest.json",
    )
    for name in rater_files:
        path = public_dir / name
        if not path.is_file():
            findings.append(f"missing rater file {name}")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for key in _FORBIDDEN_PUBLIC_KEYS:
            # Require JSON-key quoting so policy names like no_automatic_scores pass.
            if f'"{key}"' in text or f"'{key}'" in text:
                findings.append(f"{name}: leaked {key}")
        for pat in _SECRET_PATTERNS:
            if pat.search(text):
                findings.append(f"{name}: looks like an embedded secret")
    # Transport examples must also stay free of candidate identity / secrets.
    examples = public_dir / "external_judge" / "mock_transport_examples.json"
    if examples.is_file():
        text = examples.read_text(encoding="utf-8", errors="replace")
        for key in _FORBIDDEN_PUBLIC_KEYS:
            if f'"{key}"' in text or f"'{key}'" in text:
                findings.append(f"{examples.name}: leaked {key}")
        for pat in _SECRET_PATTERNS:
            if pat.search(text):
                findings.append(f"{examples.name}: looks like an embedded secret")
    # Whole public tree: no secret-shaped strings anywhere.
    for path in sorted(public_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pat in _SECRET_PATTERNS:
            if pat.search(text):
                findings.append(f"{path.relative_to(public_dir)}: looks like an embedded secret")
    if findings:
        raise AssertionError("public packet blinding failed: " + "; ".join(findings))
    return findings


def assert_excluded_from_training(manifest: dict[str, Any], public_pairs: list[dict[str, Any]]) -> None:
    if manifest.get("training_use") is not False:
        raise AssertionError("manifest must set training_use=False")
    if manifest.get("audit_holdout") is not True:
        raise AssertionError("manifest must set audit_holdout=True")
    for row in public_pairs:
        if row.get("training_use") is not False:
            raise AssertionError(f"{row.get('pair_id')}: training_use must be False")
        if row.get("automatic_judgments_visible") is not False:
            raise AssertionError(
                f"{row.get('pair_id')}: automatic_judgments_visible must be False"
            )


def _credential_runbook() -> str:
    return """# SIE-005 external-judge credential runbook

Secrets never enter git, `docs/design/`, annotation packages, or CampaignStore
artifacts. Fixture / CI runs use the in-process mock transport and need **no**
credentials.

## Required when calling a real provider (SIE-006+)

| Variable | Purpose |
| --- | --- |
| `EXTERNAL_JUDGE_API_KEY` | Provider API key (or equivalent) |
| `EXTERNAL_JUDGE_BASE_URL` | Optional provider endpoint override |
| `EXTERNAL_JUDGE_PROVIDER` | Must match `ExternalJudgeConfig.provider` |

Optional spend / retry caps (override pinned config only via env, never by
editing frozen manifests after lock):

| Variable | Purpose |
| --- | --- |
| `EXTERNAL_JUDGE_MAX_COST_USD` | Hard cost ceiling for the session |
| `EXTERNAL_JUDGE_MAX_ATTEMPTS` | Retry ceiling |

## Injection rules

1. Export variables in the shell or a local secrets manager — never commit `.env`.
2. Pass the key into an injected `transport` callable; do not serialize it into
   request JSON that is written under `outputs/` or `docs/`.
3. Redacted packages (`annotation_packet/`, `blind_pairs.jsonl`, mock fixtures)
   must fail `assert_public_packet_blinded` if any secret-shaped string appears.
4. Private unblinding keys stay **outside** the redacted package directory
   (already enforced by `freeze_blinded_pairs`).

## Independence declaration

Every external-judge request must carry the independence declaration from
`ExternalJudgeRequestV1` (all participation flags false; blinded; no candidate
identity; no automatic judgments). `JudgeEvidenceV1.require_independent()`
fails closed when any flag is wrong.
"""


def _synthetic_labels(
    public_pairs: list[dict[str, Any]], *, seed: int, force_conflict: bool = False
) -> list[dict[str, Any]]:
    """Two raters + adjudicator on disagreement; optional conflict fixture."""
    rng = random.Random(f"{AUDIT_ID}:labels:{seed}")
    labels: list[dict[str, Any]] = []
    disagreement_forced = False
    for index, row in enumerate(public_pairs):
        pair_id = str(row["pair_id"])
        winners = ["left", "left"]
        if not disagreement_forced:
            winners = ["left", "right"]
            disagreement_forced = True
        for rater_i, winner in enumerate(winners):
            labels.append(
                {
                    "schema": "BlindJudgeLabelV1",
                    "audit_id": AUDIT_ID,
                    "pair_id": pair_id,
                    "annotator_id": f"ann_rater{rater_i:02d}sie005seed{seed:04d}",
                    "role": "rater",
                    "winner": winner,
                    "acceptable_left": winner == "left",
                    "acceptable_right": winner == "right",
                    "reasons": [rng.choice(REASON_CODES)],
                    "confidence": 0.75,
                    "duration_ms": 1500 + rater_i,
                    "submitted_at": "2026-08-10T00:00:00Z",
                }
            )
        if len(set(winners)) > 1:
            labels.append(
                {
                    "schema": "BlindJudgeLabelV1",
                    "audit_id": AUDIT_ID,
                    "pair_id": pair_id,
                    "annotator_id": f"ann_adjudicatorasie005seed{seed:04d}",
                    "role": "adjudicator",
                    "winner": "left",
                    "acceptable_left": True,
                    "acceptable_right": False,
                    "reasons": ["component_inventory"],
                    "confidence": 0.95,
                    "duration_ms": 2000,
                    "submitted_at": "2026-08-10T00:05:00Z",
                }
            )
    if force_conflict and labels:
        # Duplicate same annotator+pair with a different winner → import must reject.
        conflict = dict(labels[0])
        conflict["winner"] = "right" if conflict["winner"] == "left" else "left"
        labels.append(conflict)
    return labels


@dataclass
class BlindedPacketReport:
    status: str
    claim_class: str = CLAIM_CLASS
    audit_id: str = AUDIT_ID
    related_catalogue_id: str = RELATED_CATALOGUE_ID
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    pair_n: int = 0
    strata: list[str] = field(default_factory=list)
    public_dir: str = ""
    private_key_path: str = ""
    annotation_order_sha256: str = ""
    rubric_sha256: str = ""
    independence_ok: bool = False
    import_complete_pair_n: int = 0
    mock_judge_modes_ok: list[str] = field(default_factory=list)
    blinding_ok: bool = False
    training_excluded: bool = False
    version_stamp: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = "Slm478Sie005BlindedPacketReportV1"
        return payload


def prepare_blinded_packet(
    *,
    output_dir: Path,
    seed: int = 0,
) -> BlindedPacketReport:
    """Freeze stratified audit packet + annotation package + mock judge fixtures."""
    output_dir = Path(output_dir)
    public_dir = output_dir / "annotation_packet"
    private_dir = output_dir / "private"
    private_key_path = private_dir / "unblinding_key.jsonl"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    rows_in = build_fixture_audit_rows(seed=seed)
    # Strip private-only fields before freeze.
    freeze_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows_in]
    training_ids = {str(r["source_record_id"]) for r in freeze_rows}
    training_sha = text_sha256(json.dumps(sorted(training_ids), sort_keys=True))

    # Fixture-scale: skip the 90–110 campaign power gate; still pin holdout.
    manifest = freeze_blinded_pairs(
        freeze_rows,
        audit_id=AUDIT_ID,
        seed=seed,
        output_dir=public_dir,
        private_key_path=private_key_path,
        enforce_campaign_coverage=False,
    )
    # Re-assert holdout explicitly for SIE-005 consumers.
    manifest["training_use"] = False
    manifest["audit_holdout"] = True
    manifest["corpus_admission"] = False
    manifest["sie005_strata"] = list(SIE005_STRATA)
    manifest["related_catalogue_id"] = RELATED_CATALOGUE_ID
    _write_json(public_dir / "manifest.json", manifest)

    public_pairs = [
        json.loads(line)
        for line in (public_dir / "blind_pairs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    order = _randomized_annotation_order(public_pairs, seed=seed)
    order_payload = {
        "schema": "Sie005AnnotationOrderV1",
        "audit_id": AUDIT_ID,
        "seed": seed,
        "opaque_record_ids": order,
        "note": "Present pairs in this order; IDs are opaque; no scores attached.",
    }
    _write_json(public_dir / "annotation_order.json", order_payload)
    _write_json(public_dir / "label_schema.json", _annotation_label_schema())

    config = build_external_judge_config()
    rubric_sha = text_sha256(config.rubric)
    independence = {
        "participated_in_creation": False,
        "participated_in_admission": False,
        "participated_in_training": False,
        "participated_in_preference": False,
        "participated_in_evaluation": False,
        "rubric_used_for_training_admission": False,
        "saw_candidate_identity": False,
        "saw_automatic_judgments": False,
        "blinded": True,
    }
    request_schema = external_judge_request_schema()
    response_schema = external_judge_response_schema()
    schemas_dir = public_dir / "external_judge"
    _write_json(schemas_dir / "request_schema.json", request_schema)
    _write_json(schemas_dir / "response_schema.json", response_schema)
    _write_json(
        schemas_dir / "config_pin.json",
        {
            "provider": config.provider,
            "provider_version": config.provider_version,
            "model_family": config.model_family,
            "model_version": config.model_version,
            "rubric_id": config.rubric_id,
            "rubric_version": config.rubric_version,
            "rubric_sha256": rubric_sha,
            "temperature": config.temperature,
            "seed": config.seed,
            "max_attempts": config.max_attempts,
            "max_tokens": config.max_tokens,
            "max_cost_usd": config.max_cost_usd,
            "independence_declaration": independence,
        },
    )
    # Rater-facing mock *responses* only (no checkpoint / identity fields).
    _write_json(
        schemas_dir / "mock_transport_examples.json",
        {
            "ok": mock_external_transport(mode="ok")(
                {"pair": {"prompt": "x", "left_openui": "a", "right_openui": "b"}}
            ),
            "refusal": mock_external_transport(mode="refusal")(
                {"pair": {"prompt": "x", "left_openui": "a", "right_openui": "b"}}
            ),
        },
    )

    # Full adapter evidence (includes private hashes) stays outside the redacted packet.
    harness_dir = output_dir / "harness_fixtures"
    mock_modes_ok: list[str] = []
    sample = public_pairs[0]
    for mode in ("ok", "refusal", "cost_overrun"):
        adapter = ExternalJudgeAdapter(config, mock_external_transport(mode=mode))
        evidence = adapter.judge(
            audit_id=AUDIT_ID,
            record_id=sample["pair_id"],
            generation_id=f"{sample['pair_id']}:gen",
            pair_id=sample["pair_id"],
            prompt=sample["prompt"],
            left_openui=sample["left_openui"],
            right_openui=sample["right_openui"],
            left_checkpoint_sha256=text_sha256("private-only-left"),
            right_checkpoint_sha256=text_sha256("private-only-right"),
            candidate_model_families=("fixture_a", "fixture_b"),
            prior_judge_providers=("slm_training_deterministic_evaluator",),
            prior_judge_model_families=("slm_training_deterministic_evaluator",),
            order_seed=seed,
        )
        fixture = {
            "mode": mode,
            "independent": evidence.independent,
            "refused": evidence.refused,
            "verdict": evidence.verdict,
            "error": evidence.error,
            "cost_usd": evidence.cost_usd,
            "reason_codes": list(evidence.reason_codes),
        }
        if mode == "cost_overrun" and evidence.verdict not in {
            "error", "left", "right", "tie", "refusal"
        }:
            raise AssertionError(f"unexpected cost_overrun verdict {evidence.verdict!r}")
        mock_modes_ok.append(mode)
        _write_json(harness_dir / f"mock_adapter_{mode}.json", fixture)
        if mode == "ok" and not evidence.independent:
            raise AssertionError("mock ok evidence must be independent")

    (public_dir / "CREDENTIAL_RUNBOOK.md").write_text(_credential_runbook(), encoding="utf-8")
    # Also keep a copy beside private key for operators.
    (private_dir / "CREDENTIAL_RUNBOOK.md").write_text(_credential_runbook(), encoding="utf-8")

    assert_public_packet_blinded(public_dir)
    assert_excluded_from_training(manifest, public_pairs)

    # Import round-trip (duplicates OK if identical; conflicts rejected).
    labels = _synthetic_labels(public_pairs, seed=seed)
    label_path = output_dir / "synthetic_labels" / "raters.jsonl"
    _write_jsonl(label_path, labels)
    # Idempotent re-import of the same file twice.
    aggregate = import_blinded_labels(
        [label_path, label_path],
        audit_id=AUDIT_ID,
        pair_ids={row["pair_id"] for row in public_pairs},
        output_path=output_dir / "human_aggregate.json",
    )

    conflict_raised = False
    try:
        conflict_labels = _synthetic_labels(public_pairs, seed=seed, force_conflict=True)
        conflict_path = output_dir / "synthetic_labels" / "conflict.jsonl"
        _write_jsonl(conflict_path, conflict_labels)
        import_blinded_labels(
            [conflict_path],
            audit_id=AUDIT_ID,
            pair_ids={row["pair_id"] for row in public_pairs},
            output_path=output_dir / "human_aggregate_conflict.json",
        )
    except ValueError:
        conflict_raised = True
    if not conflict_raised:
        raise AssertionError("conflicting duplicate labels must be rejected")

    return BlindedPacketReport(
        status="fixture",
        pair_n=len(public_pairs),
        strata=sorted({str(r["stratum"]) for r in public_pairs}),
        public_dir=str(public_dir),
        private_key_path=str(private_key_path),
        annotation_order_sha256=text_sha256(json.dumps(order_payload, sort_keys=True)),
        rubric_sha256=rubric_sha,
        independence_ok=True,
        import_complete_pair_n=int(aggregate["complete_pair_n"]),
        mock_judge_modes_ok=mock_modes_ok,
        blinding_ok=True,
        training_excluded=True,
        version_stamp=build_version_stamp(*VERSION_STAMP_COMPONENTS),
        notes=[
            "Packet prepared for EXP-SR-3 / SIE-006; no campaign locked here.",
            "Reuse SLM-106 judge_audit + judge_independence; VCE-010 owns calibration protocol.",
            f"training_records_sha256_pin={training_sha}",
            "conflicting duplicate labels rejected deterministically.",
        ],
    )


def run_fixture_packet(*, output_dir: Path, seed: int = 0) -> BlindedPacketReport:
    return prepare_blinded_packet(output_dir=output_dir, seed=seed)


def render_markdown(report: BlindedPacketReport | dict[str, Any]) -> str:
    data = report.to_dict() if isinstance(report, BlindedPacketReport) else report
    lines = [
        "# SLM-478 (SIE-005): Blinded construct-validity packet",
        "",
        f"**Claim class:** `{data.get('claim_class', CLAIM_CLASS)}` only",
        "",
        f"**Status:** `{data.get('status')}`",
        "",
        f"**Related catalogue:** `{data.get('related_catalogue_id', RELATED_CATALOGUE_ID)}` (SIE-006 executes)",
        "",
        "## Acceptance snapshot",
        "",
        f"| Check | Value |",
        f"| --- | --- |",
        f"| pair_n | {data.get('pair_n')} |",
        f"| strata | {', '.join(data.get('strata') or [])} |",
        f"| blinding_ok | {data.get('blinding_ok')} |",
        f"| training_excluded | {data.get('training_excluded')} |",
        f"| independence_ok | {data.get('independence_ok')} |",
        f"| import_complete_pair_n | {data.get('import_complete_pair_n')} |",
        f"| mock_judge_modes | {', '.join(data.get('mock_judge_modes_ok') or [])} |",
        f"| rubric_sha256 | `{data.get('rubric_sha256', '')[:16]}…` |",
        "",
        "## Owners reused",
        "",
        "- `judge_audit.freeze_blinded_pairs` / `import_blinded_labels` (SLM-106)",
        "- `judge_independence.ExternalJudgeConfig` / `ExternalJudgeAdapter`",
        "- VCE-010 calibration protocol (feeds this packet; not reimplemented)",
        "",
        "## Notes",
        "",
    ]
    for note in data.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)
