"""KERN-08 / SLM-531: mechanism triggers, no-effect, and corpus-local dominance.

Mirrors ``LeverProofLean.MechanismTrigger``. Autoresearch admission may emit
no-effect / dominance certificates only when every locked-corpus row has
**complete** trigger evidence and a discharged necessary-trigger law.
Unknown activation never becomes no-effect. Results stay corpus-local.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

MECHANISM_TRIGGER_SCHEMA = "mechanism_trigger_no_effect/v1"
FIXTURE_SCHEMA = "mechanism_trigger_fixtures/v1"
CERTIFICATE_SCHEMA = "mechanism_no_effect_certificate/v1"
DOMINANCE_SCHEMA = "mechanism_dominance_certificate/v1"

THEOREM_NECESSARY = "MechanismTrigger.no_effect_of_necessary_and_absent"
THEOREM_SINGLETON = "MechanismTrigger.singleton_bypass_output_change_implies_trigger"
THEOREM_CLOSURE = "MechanismTrigger.closure_removal_output_change_implies_trigger"
THEOREM_CACHE = "MechanismTrigger.cache_reuse_output_change_implies_trigger"
THEOREM_FORCED = "MechanismTrigger.forced_span_output_change_implies_trigger"
THEOREM_UNKNOWN_BLOCKS = "MechanismTrigger.unknown_activation_blocks_no_effect_certificate"
THEOREM_UNSAFE = "MechanismTrigger.unconstrained_rerank_no_safe_trigger_theorem"

_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "formal"
    / "mechanism_trigger_fixtures.v1.json"
)


class TriggerEvidence(str, Enum):
    ABSENT = "absent"
    PRESENT = "present"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class NamedCostModel:
    id: str
    version: int = 1

    @property
    def tag(self) -> str:
        return f"{self.id}.v{self.version}"


NEURAL_FORWARD_ONLY = NamedCostModel("NeuralForwardOnlyModel", 1)
DECODE_UNIT_WORK = NamedCostModel("DecodeUnitWorkModel", 1)


@dataclass(frozen=True)
class Mechanism:
    id: str
    has_safe_trigger_theorem: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "has_safe_trigger_theorem": self.has_safe_trigger_theorem,
        }


SINGLETON_BYPASS = Mechanism("singleton_bypass", True)
CLOSURE_REMOVAL = Mechanism("closure_removal", True)
CACHE_REUSE = Mechanism("cache_reuse", True)
FORCED_SPAN = Mechanism("forced_span", True)
UNCONSTRAINED_RERANK = Mechanism("unconstrained_rerank", False)


@dataclass(frozen=True)
class Observation:
    input_id: str
    baseline_output: int
    enabled_output: int
    baseline_cost: int
    enabled_cost: int
    trigger: TriggerEvidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "baseline_output": self.baseline_output,
            "enabled_output": self.enabled_output,
            "baseline_cost": self.baseline_cost,
            "enabled_cost": self.enabled_cost,
            "trigger": self.trigger.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Observation:
        return cls(
            input_id=str(payload["input_id"]),
            baseline_output=int(payload["baseline_output"]),
            enabled_output=int(payload["enabled_output"]),
            baseline_cost=int(payload["baseline_cost"]),
            enabled_cost=int(payload["enabled_cost"]),
            trigger=TriggerEvidence(str(payload["trigger"])),
        )


def trigger_evidence_complete(trigger: TriggerEvidence) -> bool:
    """Lean ``triggerEvidenceComplete`` — unknown is incomplete."""

    return trigger is not TriggerEvidence.UNKNOWN


def outputs_differ(obs: Observation) -> bool:
    return obs.enabled_output != obs.baseline_output


def outputs_equal(obs: Observation) -> bool:
    return obs.enabled_output == obs.baseline_output


def necessary_trigger(obs: Observation) -> bool:
    """Executable discharge: if outputs differ, trigger must be present.

    Rows with ``unknown`` fail (effect with incomplete evidence is not a
    discharged necessary-trigger law).
    """

    if not outputs_differ(obs):
        return True
    return obs.trigger is TriggerEvidence.PRESENT


def no_effect_from_absent(obs: Observation) -> bool:
    if obs.trigger is not TriggerEvidence.ABSENT:
        return True
    return outputs_equal(obs)


def dominates_locally(obs: Observation) -> bool:
    return outputs_equal(obs) and obs.enabled_cost >= obs.baseline_cost


def _require_wf_absent_equals(
    *,
    trigger_bit: bool,
    baseline_output: int,
    enabled_output: int,
) -> None:
    if not trigger_bit and baseline_output != enabled_output:
        raise ValueError(
            "well-formedness violated: absent trigger requires equal outputs"
        )


def observation_from_singleton_bypass(
    *,
    input_id: str,
    is_singleton: bool,
    baseline_output: int,
    enabled_output: int,
    baseline_cost: int,
    enabled_cost: int,
) -> Observation:
    _require_wf_absent_equals(
        trigger_bit=is_singleton,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
    )
    return Observation(
        input_id=input_id,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
        baseline_cost=baseline_cost,
        enabled_cost=enabled_cost,
        trigger=TriggerEvidence.PRESENT if is_singleton else TriggerEvidence.ABSENT,
    )


def observation_from_closure_removal(
    *,
    input_id: str,
    removable_nonempty: bool,
    baseline_output: int,
    enabled_output: int,
    baseline_cost: int,
    enabled_cost: int,
) -> Observation:
    _require_wf_absent_equals(
        trigger_bit=removable_nonempty,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
    )
    return Observation(
        input_id=input_id,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
        baseline_cost=baseline_cost,
        enabled_cost=enabled_cost,
        trigger=(
            TriggerEvidence.PRESENT if removable_nonempty else TriggerEvidence.ABSENT
        ),
    )


def observation_from_cache_reuse(
    *,
    input_id: str,
    cache_hit: bool,
    baseline_output: int,
    enabled_output: int,
    baseline_cost: int,
    enabled_cost: int,
) -> Observation:
    _require_wf_absent_equals(
        trigger_bit=cache_hit,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
    )
    return Observation(
        input_id=input_id,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
        baseline_cost=baseline_cost,
        enabled_cost=enabled_cost,
        trigger=TriggerEvidence.PRESENT if cache_hit else TriggerEvidence.ABSENT,
    )


def observation_from_forced_span(
    *,
    input_id: str,
    forced_nonempty: bool,
    baseline_output: int,
    enabled_output: int,
    baseline_cost: int,
    enabled_cost: int,
) -> Observation:
    _require_wf_absent_equals(
        trigger_bit=forced_nonempty,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
    )
    return Observation(
        input_id=input_id,
        baseline_output=baseline_output,
        enabled_output=enabled_output,
        baseline_cost=baseline_cost,
        enabled_cost=enabled_cost,
        trigger=TriggerEvidence.PRESENT if forced_nonempty else TriggerEvidence.ABSENT,
    )


@dataclass(frozen=True)
class NoEffectCertificateV1:
    """Corpus-local no-effect certificate for autoresearch admission."""

    schema: str
    mechanism_id: str
    cost_model: NamedCostModel
    corpus_id: str
    theorem_ref: str
    corpus: tuple[Observation, ...]
    claim_class: str
    generalizes_beyond_corpus: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "mechanism_id": self.mechanism_id,
            "cost_model": self.cost_model.tag,
            "corpus_id": self.corpus_id,
            "theorem_ref": self.theorem_ref,
            "corpus": [row.to_dict() for row in self.corpus],
            "claim_class": self.claim_class,
            "generalizes_beyond_corpus": self.generalizes_beyond_corpus,
            "all_outputs_equal": all(outputs_equal(o) for o in self.corpus),
        }


@dataclass(frozen=True)
class DominanceCertificateV1:
    no_effect: NoEffectCertificateV1
    schema: str = DOMINANCE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        payload = self.no_effect.to_dict()
        payload["schema"] = self.schema
        payload["dominates_locally"] = all(
            dominates_locally(o) for o in self.no_effect.corpus
        )
        return payload


class CertificateError(ValueError):
    """Fail-closed certificate construction."""


def emit_no_effect_certificate(
    *,
    mechanism: Mechanism,
    cost_model: NamedCostModel,
    corpus_id: str,
    corpus: Sequence[Observation],
    theorem_ref: str = THEOREM_NECESSARY,
    claim_class: str = "fixture",
) -> NoEffectCertificateV1:
    """Build a no-effect certificate; fails closed on incomplete evidence."""

    if not mechanism.has_safe_trigger_theorem:
        raise CertificateError(
            f"mechanism {mechanism.id!r} has no safe trigger theorem"
        )
    if not corpus:
        raise CertificateError("corpus must be non-empty")
    rows = tuple(corpus)
    for obs in rows:
        if not trigger_evidence_complete(obs.trigger):
            raise CertificateError(
                f"unknown trigger evidence on {obs.input_id!r} never becomes no-effect"
            )
        if not necessary_trigger(obs):
            raise CertificateError(
                f"necessary-trigger law fails on {obs.input_id!r}"
            )
        if obs.trigger is not TriggerEvidence.ABSENT:
            raise CertificateError(
                f"no-effect certificate requires trigger absent on {obs.input_id!r}"
            )
        if not outputs_equal(obs):
            raise CertificateError(
                f"absent trigger but outputs differ on {obs.input_id!r}"
            )
    return NoEffectCertificateV1(
        schema=CERTIFICATE_SCHEMA,
        mechanism_id=mechanism.id,
        cost_model=cost_model,
        corpus_id=corpus_id,
        theorem_ref=theorem_ref,
        corpus=rows,
        claim_class=claim_class,
        generalizes_beyond_corpus=False,
    )


def emit_dominance_certificate(
    *,
    mechanism: Mechanism,
    cost_model: NamedCostModel,
    corpus_id: str,
    corpus: Sequence[Observation],
    theorem_ref: str = THEOREM_NECESSARY,
    claim_class: str = "fixture",
) -> DominanceCertificateV1:
    """No-effect plus treatment cost ≥ baseline on every locked row."""

    no_effect = emit_no_effect_certificate(
        mechanism=mechanism,
        cost_model=cost_model,
        corpus_id=corpus_id,
        corpus=corpus,
        theorem_ref=theorem_ref,
        claim_class=claim_class,
    )
    for obs in no_effect.corpus:
        if obs.enabled_cost < obs.baseline_cost:
            raise CertificateError(
                f"dominance requires enabled_cost >= baseline_cost on {obs.input_id!r}"
            )
    return DominanceCertificateV1(no_effect=no_effect)


def unconstrained_rerank_counterexample() -> Observation:
    """Mechanism with no safe trigger theorem — outputs may differ freely."""

    return Observation(
        input_id="counterexample/unconstrained_rerank",
        baseline_output=0,
        enabled_output=1,
        baseline_cost=1,
        enabled_cost=1,
        trigger=TriggerEvidence.UNKNOWN,
    )


def unknown_activation_never_no_effect() -> bool:
    """Acceptance: unknown evidence is rejected by certificate emission."""

    obs = Observation(
        input_id="counterexample/unknown_activation",
        baseline_output=1,
        enabled_output=1,
        baseline_cost=1,
        enabled_cost=1,
        trigger=TriggerEvidence.UNKNOWN,
    )
    try:
        emit_no_effect_certificate(
            mechanism=SINGLETON_BYPASS,
            cost_model=NEURAL_FORWARD_ONLY,
            corpus_id="counterexample",
            corpus=[obs],
        )
    except CertificateError:
        return True
    return False


def load_fixture_payload() -> dict[str, Any]:
    payload = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError(f"unexpected fixture schema: {payload.get('schema')!r}")
    return payload


def certified_fixture_rows() -> list[dict[str, Any]]:
    """Evaluate frozen fixtures into live certificate / counterexample rows."""

    payload = load_fixture_payload()
    rows: list[dict[str, Any]] = []
    for case in payload["cases"]:
        kind = str(case["kind"])
        mech = Mechanism(
            id=str(case["mechanism_id"]),
            has_safe_trigger_theorem=bool(case["has_safe_trigger_theorem"]),
        )
        model = NamedCostModel(
            id=str(case.get("cost_model", NEURAL_FORWARD_ONLY.id)),
            version=int(case.get("cost_model_version", 1)),
        )
        obs = Observation.from_dict(case["observation"])
        entry: dict[str, Any] = {
            "id": case["id"],
            "kind": kind,
            "mechanism": mech.to_dict(),
            "observation": obs.to_dict(),
            "necessary_trigger": necessary_trigger(obs),
            "trigger_evidence_complete": trigger_evidence_complete(obs.trigger),
            "outputs_equal": outputs_equal(obs),
            "dominates_locally": dominates_locally(obs),
        }
        if kind == "no_effect":
            cert = emit_no_effect_certificate(
                mechanism=mech,
                cost_model=model,
                corpus_id=str(case["corpus_id"]),
                corpus=[obs],
                theorem_ref=str(case.get("theorem_ref", THEOREM_NECESSARY)),
            )
            entry["certificate"] = cert.to_dict()
        elif kind == "dominance":
            cert_d = emit_dominance_certificate(
                mechanism=mech,
                cost_model=model,
                corpus_id=str(case["corpus_id"]),
                corpus=[obs],
                theorem_ref=str(case.get("theorem_ref", THEOREM_NECESSARY)),
            )
            entry["certificate"] = cert_d.to_dict()
        elif kind == "counterexample":
            entry["expected_reject"] = True
            if case.get("expect_certificate_error"):
                try:
                    emit_no_effect_certificate(
                        mechanism=mech,
                        cost_model=model,
                        corpus_id=str(case.get("corpus_id", "counterexample")),
                        corpus=[obs],
                    )
                    entry["certificate_error"] = False
                except CertificateError as exc:
                    entry["certificate_error"] = True
                    entry["error"] = str(exc)
        else:
            raise ValueError(f"unknown fixture kind: {kind!r}")
        rows.append(entry)
    return rows


__all__ = [
    "CACHE_REUSE",
    "CERTIFICATE_SCHEMA",
    "CLOSURE_REMOVAL",
    "DECODE_UNIT_WORK",
    "DOMINANCE_SCHEMA",
    "FIXTURE_SCHEMA",
    "FORCED_SPAN",
    "MECHANISM_TRIGGER_SCHEMA",
    "NEURAL_FORWARD_ONLY",
    "SINGLETON_BYPASS",
    "THEOREM_CACHE",
    "THEOREM_CLOSURE",
    "THEOREM_FORCED",
    "THEOREM_NECESSARY",
    "THEOREM_SINGLETON",
    "THEOREM_UNKNOWN_BLOCKS",
    "THEOREM_UNSAFE",
    "UNCONSTRAINED_RERANK",
    "CertificateError",
    "DominanceCertificateV1",
    "Mechanism",
    "NamedCostModel",
    "NoEffectCertificateV1",
    "Observation",
    "TriggerEvidence",
    "certified_fixture_rows",
    "dominates_locally",
    "emit_dominance_certificate",
    "emit_no_effect_certificate",
    "load_fixture_payload",
    "necessary_trigger",
    "no_effect_from_absent",
    "observation_from_cache_reuse",
    "observation_from_closure_removal",
    "observation_from_forced_span",
    "observation_from_singleton_bypass",
    "outputs_differ",
    "outputs_equal",
    "trigger_evidence_complete",
    "unconstrained_rerank_counterexample",
    "unknown_activation_never_no_effect",
]
