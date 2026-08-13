"""Bounded prompt-intervention diagnostics.

Fixture wiring only. A disposition such as ``objective_sampler_mismatch``
is a label, not a closed experiment and not a capability certificate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.harness_core.lineage.records import content_sha
from slm_training.versioning import build_version_stamp

SCHEMA_VERSION = "learnability_diagnostics/v1"
CLAIM_CLASS = "fixture_wiring"
STAMP_COMPONENTS = ("evals.scoring",)
PROMPT_RECORD_LIMIT = 4
ARTIFACT_NAME = "learnability_diagnostics.json"

_GENERIC = (
    "ui",
    "screen",
    "layout",
    "page",
    "interface",
    "build",
    "create",
    "generate",
    "design",
    "openui",
)
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class PromptIntervention:
    name: str
    prompt: str
    group_id: str
    preserves_semantics: bool
    target_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt": self.prompt,
            "group_id": self.group_id,
            "preserves_semantics": self.preserves_semantics,
            "target_id": self.target_id,
        }


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def make_interventions(
    prompt: str,
    *,
    group_id: str,
    target_id: str,
) -> tuple[PromptIntervention, ...]:
    """Same-target cue interventions. Never copy the gold target into metadata."""

    words = prompt.split()
    shuffled = " ".join(reversed(words)) if len(words) > 1 else prompt
    ablated = " ".join(word for word in words if word.lower().strip(".,;:") not in _GENERIC)
    wrapper = f"Note: {prompt}"
    no_noun = " ".join(
        word
        for word in words
        if word.lower().strip(".,;:") not in {"ui", "screen", "layout", "page", "interface"}
    )
    schema = f"schema request: {prompt}"
    return (
        PromptIntervention("correct", prompt, group_id, True, target_id),
        PromptIntervention("shuffled", shuffled, group_id, True, None),
        PromptIntervention("cue_ablation", ablated or prompt, group_id, True, None),
        PromptIntervention("wrapper_substitution", wrapper, group_id, True, None),
        PromptIntervention("no_format_noun", no_noun or prompt, group_id, True, None),
        PromptIntervention("grammar_schema_control", schema, group_id, False, None),
    )


def interpret_probe(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Typed disposition. Incomplete/timeout stays incomplete, never a zero."""

    if metrics.get("incomplete") or metrics.get("timeout"):
        return {
            "disposition": "incomplete",
            "claim_class": CLAIM_CLASS,
            "reason": "timeout_or_incomplete_is_not_a_quality_zero",
        }
    train_exact = metrics.get("train_exact")
    shuffled_same = metrics.get("shuffled_output_equals_correct")
    ablation_changes = metrics.get("ablation_changes_output")
    partial_ok = metrics.get("partial_mask_ok")
    all_mask_ok = metrics.get("all_mask_ok")
    raw_rank_ok = metrics.get("raw_legal_rank_useful")
    constrained_regressed = metrics.get("constrained_choice_regressed")
    holdout_fail = metrics.get("topology_holdout_fail")
    if train_exact is False:
        diagnosis = "foundational_model_or_tokenization"
    elif shuffled_same is True:
        diagnosis = "conditioning_or_data_alignment"
    elif ablation_changes is True:
        diagnosis = "shortcut_dependence"
    elif partial_ok and all_mask_ok is False:
        diagnosis = "objective_sampler_mismatch"
    elif raw_rank_ok and constrained_regressed:
        diagnosis = "decoder_policy"
    elif train_exact and holdout_fail:
        diagnosis = "coverage_generalization"
    else:
        diagnosis = "unresolved"
    return {
        "disposition": diagnosis,
        "claim_class": CLAIM_CLASS,
        "capability_certificate": False,
        "metrics": dict(metrics),
    }


def _suite_jsonl(test_dir: Path, suite: str) -> Path:
    from slm_training.data.store import DataStore

    resolved = DataStore().resolve_path("eval", test_dir)
    manifest_path = resolved / "manifest.json"
    if manifest_path.is_file():
        try:
            declared = (
                json.loads(manifest_path.read_text(encoding="utf-8")).get("suites") or {}
            ).get(suite)
        except (OSError, json.JSONDecodeError, TypeError):
            declared = None
        if declared:
            path = Path(declared)
            if path.is_file():
                return path
    return resolved / "suites" / suite / "records.jsonl"


def load_suite_prompt_records(
    test_dir: Path | str | None,
    suites: Sequence[str],
    *,
    limit: int = PROMPT_RECORD_LIMIT,
) -> tuple[list[dict[str, str]], list[str]]:
    """First N suite ``prompt`` fields. Never invent prompts from metric names."""

    if test_dir is None:
        return [], ["test_dir is missing"]
    records: list[dict[str, str]] = []
    reasons: list[str] = []
    for suite in suites:
        try:
            path = _suite_jsonl(Path(test_dir), suite)
        except (OSError, ValueError) as exc:
            reasons.append(f"{suite}: {exc}")
            continue
        if not path.is_file():
            reasons.append(f"{suite}: missing suite records at {path}")
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            reasons.append(f"{suite}: {exc}")
            continue
        taken = 0
        for line in lines:
            if taken >= limit:
                break
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                continue
            records.append(
                {
                    "id": str(row.get("id") or f"{suite}-{taken}"),
                    "prompt": prompt,
                    "group_id": suite,
                }
            )
            taken += 1
        if taken == 0:
            reasons.append(f"{suite}: no prompt fields in {path}")
    return records, reasons


def diagnose_records(
    records: Sequence[Mapping[str, Any]],
    *,
    probe_metrics: Mapping[str, Any] | None = None,
    unavailable_reasons: Sequence[str] | None = None,
) -> dict[str, Any]:
    interventions = []
    for index, record in enumerate(records):
        group = str(record.get("group_id") or f"g{index}")
        target_id = str(record.get("id") or f"t{index}")
        prompt = str(record.get("prompt") or "")
        items = make_interventions(prompt, group_id=group, target_id=target_id)
        interventions.extend(item.to_dict() for item in items)
        for item in items:
            if item.name != "correct":
                assert item.target_id is None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "claim_class": CLAIM_CLASS,
        "capability_certificate": False,
        "version_stamp": build_version_stamp(*STAMP_COMPONENTS),
        "interventions": interventions,
        "unavailable_reasons": list(unavailable_reasons or ()),
        "interpretation": interpret_probe(probe_metrics or {}),
        "digest": hashlib.sha256(
            repr(tuple(item["name"] for item in interventions)).encode("utf-8")
        ).hexdigest(),
    }
    payload["sha256"] = content_sha(
        {key: value for key, value in payload.items() if key != "sha256"}
    )
    return payload


def write_learnability_diagnostics(
    run_dir: Path | str,
    *,
    test_dir: Path | str | None,
    suites: Sequence[str],
    probe_metrics: Mapping[str, Any] | None = None,
) -> Path:
    records, reasons = load_suite_prompt_records(test_dir, suites)
    payload = diagnose_records(
        records,
        probe_metrics=probe_metrics,
        unavailable_reasons=reasons,
    )
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ARTIFACT_NAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
