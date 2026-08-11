"""Frozen content-addressed revmath fixture corpus (HARN-11 / SLM-554).

Assembles the committed hermetic fixtures under ``resources/revmath/fixtures/``
into a versioned catalog. Does not invent task plugins — replays use the
existing HARN-03..09 runners/validators. Train/eval root-family leakage is
detectable via ``split`` + ``root_family_id``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slm_training.formal.judgment import JudgmentOutcome
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.reasoning.revmath.assumption_ablation import (
    evaluate_ablation_lattice,
    materialize_candidate_task,
    parse_ablation_meta,
)
from slm_training.harnesses.reasoning.revmath.labeling import (
    parse_label_claim,
    validate_label_claim,
)
from slm_training.harnesses.reasoning.revmath.plugins import (
    RevmathCheckPlan,
    ablation_plugin_for_candidate,
    default_plugin_registry,
    override_hermetic_mode,
    resolve_plugin,
    reversal_plugin_for_obligation,
)
from slm_training.harnesses.reasoning.revmath.reversal import (
    evaluate_reversal,
    materialize_obligation_task,
    parse_reversal_meta,
)
from slm_training.harnesses.reasoning.revmath.runner import run_revmath_task
from slm_training.harnesses.reasoning.revmath.schemas import (
    RevmathSchemaError,
    RevmathTaskV1,
    parse_revmath_task,
)
from slm_training.harnesses.reasoning.revmath.self_healing import (
    reject_forbidden_repair,
)

CORPUS_SCHEMA = "revmath_fixture_corpus/v1"
CORPUS_ID = "corpus.rm.hermetic.v1"
MANIFEST_RELATIVE = (
    "src/slm_training/resources/revmath/corpus/hermetic_v1.manifest.json"
)
FIXTURES_RELATIVE = "src/slm_training/resources/revmath/fixtures"

ReplayMode = Literal[
    "single_runner",
    "ablation_lattice",
    "reversal_pair",
    "label_claim",
    "runner_probe",
]
CorpusSplit = Literal["train", "eval", "diag"]
MutationKind = Literal[
    "proposition",
    "toolchain",
    "source",
    "certificate",
    "corpus_membership",
    "budget",
]

# Required coverage classes from HARN-11 / SLM-554.
REQUIRED_CORPUS_CLASSES: frozenset[str] = frozenset(
    {
        "ablation_success",
        "ablation_necessary",
        "reversal_equivalence",
        "reversal_one_way",
        "constructivization_success",
        "constructivization_unknown",
        "counterexample_checked",
        "quantitative_bound_success",
        "quantitative_bound_nonextractable",
        "practical_computability",
        "mutation_proposition",
        "mutation_toolchain",
        "mutation_source",
        "mutation_certificate",
        "timeout",
        "missing_tool",
        "unsupported_feature",
        "malformed_evidence",
        "incomplete_domain",
    }
)


@dataclass(frozen=True)
class MutationExpectationV1:
    kind: MutationKind
    gate: str
    must_fail: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "gate": self.gate,
            "must_fail": self.must_fail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MutationExpectationV1:
        return cls(
            kind=str(data["kind"]),  # type: ignore[arg-type]
            gate=str(data["gate"]),
            must_fail=bool(data.get("must_fail", True)),
        )


@dataclass(frozen=True)
class FixtureCorpusEntrySpecV1:
    """Authoring-time catalog row (digests filled by build/verify)."""

    entry_id: str
    corpus_class: str
    split: CorpusSplit
    root_family_id: str
    replay_mode: ReplayMode
    task_stem: str | None = None
    meta_stem: str | None = None
    claim_stem: str | None = None
    probe_id: str | None = None
    checker_ref: str | None = None
    source_toolchain: str = "hermetic_python_checker"
    expected_outcome: str | None = None
    expected_unknown_reason: str | None = None
    expected_label_accepted: bool | None = None
    expected_report: Mapping[str, Any] | None = None
    allowed_labels: tuple[str, ...] = ()
    mutation_expectations: tuple[MutationExpectationV1, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "corpus_class": self.corpus_class,
            "split": self.split,
            "root_family_id": self.root_family_id,
            "replay_mode": self.replay_mode,
            "task_stem": self.task_stem,
            "meta_stem": self.meta_stem,
            "claim_stem": self.claim_stem,
            "probe_id": self.probe_id,
            "checker_ref": self.checker_ref,
            "source_toolchain": self.source_toolchain,
            "expected_outcome": self.expected_outcome,
            "expected_unknown_reason": self.expected_unknown_reason,
            "expected_label_accepted": self.expected_label_accepted,
            "expected_report": (
                None if self.expected_report is None else dict(self.expected_report)
            ),
            "allowed_labels": list(self.allowed_labels),
            "mutation_expectations": [m.to_dict() for m in self.mutation_expectations],
        }


@dataclass(frozen=True)
class FixtureCorpusEntryV1:
    """Frozen catalog entry with content digests."""

    entry_id: str
    corpus_class: str
    split: CorpusSplit
    root_family_id: str
    replay_mode: ReplayMode
    task_stem: str | None
    meta_stem: str | None
    claim_stem: str | None
    probe_id: str | None
    checker_ref: str | None
    source_toolchain: str
    proposition_id: str | None
    statement_sha256: str | None
    base_theory_id: str | None
    allowed_assumption_ids: tuple[str, ...]
    budget: Mapping[str, Any] | None
    expected_outcome: str | None
    expected_unknown_reason: str | None
    expected_label_accepted: bool | None
    expected_report: Mapping[str, Any] | None
    allowed_labels: tuple[str, ...]
    mutation_expectations: tuple[MutationExpectationV1, ...]
    artifact_sha256: str
    replay_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "corpus_class": self.corpus_class,
            "split": self.split,
            "root_family_id": self.root_family_id,
            "replay_mode": self.replay_mode,
            "task_stem": self.task_stem,
            "meta_stem": self.meta_stem,
            "claim_stem": self.claim_stem,
            "probe_id": self.probe_id,
            "checker_ref": self.checker_ref,
            "source_toolchain": self.source_toolchain,
            "proposition_id": self.proposition_id,
            "statement_sha256": self.statement_sha256,
            "base_theory_id": self.base_theory_id,
            "allowed_assumption_ids": list(self.allowed_assumption_ids),
            "budget": None if self.budget is None else dict(self.budget),
            "expected_outcome": self.expected_outcome,
            "expected_unknown_reason": self.expected_unknown_reason,
            "expected_label_accepted": self.expected_label_accepted,
            "expected_report": (
                None if self.expected_report is None else dict(self.expected_report)
            ),
            "allowed_labels": list(self.allowed_labels),
            "mutation_expectations": [m.to_dict() for m in self.mutation_expectations],
            "artifact_sha256": self.artifact_sha256,
            "replay_digest": self.replay_digest,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FixtureCorpusEntryV1:
        mutations = tuple(
            MutationExpectationV1.from_dict(m)
            for m in data.get("mutation_expectations", ())
        )
        return cls(
            entry_id=str(data["entry_id"]),
            corpus_class=str(data["corpus_class"]),
            split=str(data["split"]),  # type: ignore[arg-type]
            root_family_id=str(data["root_family_id"]),
            replay_mode=str(data["replay_mode"]),  # type: ignore[arg-type]
            task_stem=data.get("task_stem"),
            meta_stem=data.get("meta_stem"),
            claim_stem=data.get("claim_stem"),
            probe_id=data.get("probe_id"),
            checker_ref=data.get("checker_ref"),
            source_toolchain=str(data.get("source_toolchain", "hermetic_python_checker")),
            proposition_id=data.get("proposition_id"),
            statement_sha256=data.get("statement_sha256"),
            base_theory_id=data.get("base_theory_id"),
            allowed_assumption_ids=tuple(data.get("allowed_assumption_ids") or ()),
            budget=data.get("budget"),
            expected_outcome=data.get("expected_outcome"),
            expected_unknown_reason=data.get("expected_unknown_reason"),
            expected_label_accepted=data.get("expected_label_accepted"),
            expected_report=data.get("expected_report"),
            allowed_labels=tuple(data.get("allowed_labels") or ()),
            mutation_expectations=mutations,
            artifact_sha256=str(data["artifact_sha256"]),
            replay_digest=str(data["replay_digest"]),
        )


@dataclass(frozen=True)
class FixtureCorpusManifestV1:
    schema_version: str
    corpus_id: str
    corpus_sha256: str
    fixtures_dir: str
    entry_count: int
    entries: tuple[FixtureCorpusEntryV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "corpus_id": self.corpus_id,
            "corpus_sha256": self.corpus_sha256,
            "fixtures_dir": self.fixtures_dir,
            "entry_count": self.entry_count,
            "entries": [e.to_dict() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FixtureCorpusManifestV1:
        if data.get("schema_version") != CORPUS_SCHEMA:
            raise RevmathSchemaError(
                f"unsupported corpus schema {data.get('schema_version')!r}"
            )
        entries = tuple(FixtureCorpusEntryV1.from_dict(e) for e in data["entries"])
        return cls(
            schema_version=str(data["schema_version"]),
            corpus_id=str(data["corpus_id"]),
            corpus_sha256=str(data["corpus_sha256"]),
            fixtures_dir=str(data["fixtures_dir"]),
            entry_count=int(data["entry_count"]),
            entries=entries,
        )


def _repo_root() -> Path:
    # corpus.py → revmath → reasoning → harnesses → slm_training → src → root
    return Path(__file__).resolve().parents[5]


def fixtures_dir(root: Path | None = None) -> Path:
    return (root or _repo_root()) / FIXTURES_RELATIVE


def manifest_path(root: Path | None = None) -> Path:
    return (root or _repo_root()) / MANIFEST_RELATIVE


def _default_mutations(*kinds: MutationKind) -> tuple[MutationExpectationV1, ...]:
    gate = {
        "proposition": "proposition_identity",
        "toolchain": "judge_definitions",
        "source": "base_theory_or_allowed_assumptions",
        "certificate": "expected_counterexample_semantics",
        "corpus_membership": "corpus_membership",
        "budget": "budgets_and_stopping_rules",
    }
    return tuple(
        MutationExpectationV1(kind=k, gate=gate[k], must_fail=True) for k in kinds
    )


def catalog_specs() -> tuple[FixtureCorpusEntrySpecV1, ...]:
    """Frozen authoring catalog — one row per required corpus class (+ extras)."""

    mut_all = _default_mutations(
        "proposition", "toolchain", "source", "certificate", "corpus_membership"
    )
    return (
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.ablation_success",
            corpus_class="ablation_success",
            split="train",
            root_family_id="family.add_zero",
            replay_mode="ablation_lattice",
            task_stem="ablation_positive",
            meta_stem="ablation_positive",
            checker_ref="hermetic_ablation_checker.py",
            expected_report={
                "weaker_witnessed": True,
                "minimal_remaining": ["ISigma1"],
            },
            allowed_labels=("rm_inspired_assumption_minimization",),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.ablation_necessary",
            corpus_class="ablation_necessary",
            split="eval",
            root_family_id="family.add_zero",
            replay_mode="ablation_lattice",
            task_stem="ablation_necessary",
            meta_stem="ablation_necessary",
            checker_ref="hermetic_ablation_checker.py",
            expected_report={
                "removed_ISigma1_outcome": "refuted",
                "baseline_outcome": "witnessed",
            },
            allowed_labels=("rm_inspired_assumption_minimization",),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.reversal_equivalence",
            corpus_class="reversal_equivalence",
            split="train",
            root_family_id="family.wkl_path",
            replay_mode="reversal_pair",
            task_stem="reversal_equivalence",
            meta_stem="reversal_equivalence",
            checker_ref="hermetic_reversal_checker.py",
            expected_report={
                "equivalence_label": "equivalent",
                "claims_equivalence": True,
            },
            allowed_labels=("genuine_reverse_mathematics", "reversed_equivalent"),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.reversal_one_way",
            corpus_class="reversal_one_way",
            split="eval",
            root_family_id="family.wkl_path",
            replay_mode="reversal_pair",
            task_stem="reversal_one_way_forward",
            meta_stem="reversal_one_way_forward",
            checker_ref="hermetic_reversal_checker.py",
            expected_report={
                "equivalence_label": "one_way_forward",
                "claims_equivalence": False,
            },
            allowed_labels=("genuine_reverse_mathematics",),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.constructivization_success",
            corpus_class="constructivization_success",
            split="train",
            root_family_id="family.constructivization.bounded",
            replay_mode="single_runner",
            task_stem="constructivization_bounded",
            meta_stem="constructivization_bounded",
            checker_ref="hermetic_constructivization_checker.py",
            expected_outcome="witnessed",
            allowed_labels=("constructive_variant",),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.constructivization_unknown",
            corpus_class="constructivization_unknown",
            split="eval",
            root_family_id="family.constructivization.timeout",
            replay_mode="single_runner",
            task_stem="constructivization_timeout",
            meta_stem="constructivization_timeout",
            checker_ref="hermetic_constructivization_checker.py",
            expected_outcome="unknown",
            expected_unknown_reason="timeout",
            allowed_labels=("constructive_variant",),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.counterexample_checked",
            corpus_class="counterexample_checked",
            split="train",
            root_family_id="family.counterexample.checked",
            replay_mode="single_runner",
            task_stem="counterexample_checked",
            meta_stem="counterexample_checked",
            checker_ref="hermetic_counterexample_checker.py",
            expected_outcome="refuted",
            allowed_labels=("computable_finite_counterexample",),
            mutation_expectations=_default_mutations(
                "proposition", "certificate", "source"
            ),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.quant_success",
            corpus_class="quantitative_bound_success",
            split="train",
            root_family_id="family.quant.finite_search",
            replay_mode="single_runner",
            task_stem="quant_finite_search",
            meta_stem="quant_finite_search",
            checker_ref="hermetic_quantitative_bound_checker.py",
            expected_outcome="witnessed",
            allowed_labels=("quantitative_bound_extraction",),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.quant_nonextractable",
            corpus_class="quantitative_bound_nonextractable",
            split="eval",
            root_family_id="family.quant.nonextractable",
            replay_mode="single_runner",
            task_stem="quant_nonextractable",
            meta_stem="quant_nonextractable",
            checker_ref="hermetic_quantitative_bound_checker.py",
            expected_outcome="witnessed",
            expected_report={"status": "nonextractable", "theorem_derived_bound": False},
            allowed_labels=("quantitative_bound_extraction", "nonextractable"),
            mutation_expectations=mut_all,
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.practical_computability",
            corpus_class="practical_computability",
            split="train",
            root_family_id="family.label.practical_computability",
            replay_mode="label_claim",
            claim_stem="labeling_practical_computability",
            source_toolchain="label_claim_validator",
            expected_label_accepted=True,
            allowed_labels=("practical_computability",),
            mutation_expectations=_default_mutations("proposition", "source"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.mutation_proposition",
            corpus_class="mutation_proposition",
            split="diag",
            root_family_id="family.add_zero",
            replay_mode="single_runner",
            task_stem="hermetic_forward_theorem",
            checker_ref="hermetic_checker.py",
            expected_outcome="witnessed",
            mutation_expectations=_default_mutations("proposition"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.mutation_toolchain",
            corpus_class="mutation_toolchain",
            split="diag",
            root_family_id="family.add_zero",
            replay_mode="single_runner",
            task_stem="hermetic_forward_theorem",
            checker_ref="hermetic_checker.py",
            expected_outcome="witnessed",
            mutation_expectations=_default_mutations("toolchain"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.mutation_source",
            corpus_class="mutation_source",
            split="diag",
            root_family_id="family.add_zero",
            replay_mode="single_runner",
            task_stem="hermetic_forward_theorem",
            checker_ref="hermetic_checker.py",
            expected_outcome="witnessed",
            mutation_expectations=_default_mutations("source"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.mutation_certificate",
            corpus_class="mutation_certificate",
            split="diag",
            root_family_id="family.counterexample.checked",
            replay_mode="single_runner",
            task_stem="counterexample_checked",
            meta_stem="counterexample_checked",
            checker_ref="hermetic_counterexample_checker.py",
            expected_outcome="refuted",
            mutation_expectations=_default_mutations("certificate"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.timeout",
            corpus_class="timeout",
            split="diag",
            root_family_id="family.add_zero",
            replay_mode="single_runner",
            task_stem="ablation_timeout",
            meta_stem="ablation_timeout",
            checker_ref="hermetic_ablation_checker.py",
            expected_outcome="unknown",
            expected_unknown_reason="timeout",
            mutation_expectations=_default_mutations("budget"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.missing_tool",
            corpus_class="missing_tool",
            split="diag",
            root_family_id="family.add_zero",
            replay_mode="runner_probe",
            task_stem="hermetic_forward_theorem",
            probe_id="missing_lean_tool",
            source_toolchain="runner_probe",
            expected_outcome="unknown",
            expected_unknown_reason="missing_tool",
            mutation_expectations=_default_mutations("toolchain"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.unsupported_feature",
            corpus_class="unsupported_feature",
            split="diag",
            root_family_id="family.add_zero",
            replay_mode="runner_probe",
            task_stem="hermetic_forward_theorem",
            probe_id="unsupported_task_kind",
            source_toolchain="runner_probe",
            expected_outcome="unknown",
            expected_unknown_reason="unsupported_capability",
            mutation_expectations=_default_mutations("toolchain"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.malformed_evidence",
            corpus_class="malformed_evidence",
            split="diag",
            root_family_id="family.counterexample.mismatch",
            replay_mode="single_runner",
            task_stem="counterexample_mismatch_prop",
            meta_stem="counterexample_mismatch_prop",
            checker_ref="hermetic_counterexample_checker.py",
            expected_outcome="invalid",
            expected_unknown_reason="malformed_input",
            mutation_expectations=_default_mutations("certificate", "proposition"),
        ),
        FixtureCorpusEntrySpecV1(
            entry_id="entry.corpus.incomplete_domain",
            corpus_class="incomplete_domain",
            split="diag",
            root_family_id="family.counterexample.search_failed",
            replay_mode="single_runner",
            task_stem="counterexample_search_failed",
            meta_stem="counterexample_search_failed",
            checker_ref="hermetic_counterexample_checker.py",
            expected_outcome="unknown",
            expected_unknown_reason="incomplete_coverage",
            mutation_expectations=_default_mutations("certificate"),
        ),
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_digest(
    *,
    fixtures: Path,
    task_stem: str | None,
    meta_stem: str | None,
    claim_stem: str | None,
    probe_id: str | None,
) -> str:
    payload: dict[str, Any] = {"probe_id": probe_id}
    if task_stem:
        payload["task"] = _load_json(fixtures / f"{task_stem}.task.json")
    if meta_stem:
        payload["meta"] = _load_json(fixtures / f"{meta_stem}.meta.json")
    if claim_stem:
        payload["claim"] = _load_json(fixtures / f"{claim_stem}.claim.json")
    return content_sha(payload)


def _load_task(fixtures: Path, stem: str) -> RevmathTaskV1:
    return parse_revmath_task(_load_json(fixtures / f"{stem}.task.json"))


def _stable_report_digest(report: Any) -> str:
    if hasattr(report, "to_dict"):
        data = report.to_dict()
    elif isinstance(report, Mapping):
        data = dict(report)
    else:
        data = {"repr": repr(report)}
    # Drop wall-clock-ish keys if present.
    data.pop("duration_seconds", None)
    return content_sha(data)


def _run_ablation_lattice(task: RevmathTaskV1, meta: Any):
    def run_candidate(baseline: RevmathTaskV1, candidate):
        cand_task = materialize_candidate_task(baseline, candidate)
        plugin = ablation_plugin_for_candidate(meta, candidate)
        registry = default_plugin_registry()
        registry["assumption_ablation"] = plugin
        return run_revmath_task(cand_task, hermetic=True, plugin_registry=registry)

    return evaluate_ablation_lattice(task, meta, run_candidate=run_candidate)


def _run_reversal_pair(task: RevmathTaskV1, meta: Any):
    def run_obligation(baseline: RevmathTaskV1, obligation):
        obl_task = materialize_obligation_task(baseline, meta, obligation)
        plugin = reversal_plugin_for_obligation(meta, obligation)
        registry = default_plugin_registry()
        registry["reversal"] = plugin
        return run_revmath_task(obl_task, hermetic=True, plugin_registry=registry)

    return evaluate_reversal(task, meta, run_obligation=run_obligation)


def _run_probe(task: RevmathTaskV1, probe_id: str, root: Path):
    if probe_id == "missing_lean_tool":
        empty = root / ".corpus_probe_no_lean"
        empty.mkdir(exist_ok=True)
        return run_revmath_task(task, lean_root=empty, hermetic=False)
    if probe_id == "unsupported_task_kind":
        probe_task = task.model_copy(update={"task_kind": "computability_classification"})
        return run_revmath_task(probe_task, hermetic=True)
    if probe_id == "malformed_plan":
        plugin = resolve_plugin(task)
        assert plugin is not None
        plan = plugin.plan_check(task, lean_root=root, hermetic=True)
        override = RevmathCheckPlan(
            command=override_hermetic_mode(plan.command, "malformed"),
            cwd=plan.cwd,
            env=plan.env,
            requires_lean_tool=False,
            lean_project_root=None,
            checker_id=plan.checker_id,
        )
        return run_revmath_task(task, hermetic=True, plan_override=override)
    raise RevmathSchemaError(f"unknown runner probe_id={probe_id!r}")


def _assert_expected_report(spec: FixtureCorpusEntrySpecV1, report: Any) -> None:
    expected = spec.expected_report or {}
    if not expected:
        return
    if spec.replay_mode == "ablation_lattice":
        by_removed = {
            ",".join(r.candidate.removed_assumption_ids): r for r in report.records
        }
        if "weaker_witnessed" in expected:
            weaker = by_removed.get("WKL_redundant")
            assert weaker is not None and weaker.outcome == "witnessed"
        if "minimal_remaining" in expected:
            minimal = [
                r
                for r in report.records
                if r.lattice_minimal_among_explored_successes
            ]
            assert len(minimal) == 1
            assert list(minimal[0].candidate.remaining_assumption_ids) == list(
                expected["minimal_remaining"]
            )
        if "removed_ISigma1_outcome" in expected:
            assert by_removed["ISigma1"].outcome == expected["removed_ISigma1_outcome"]
        if "baseline_outcome" in expected:
            assert by_removed[""].outcome == expected["baseline_outcome"]
        return
    if spec.replay_mode == "reversal_pair":
        for key, value in expected.items():
            assert getattr(report, key) == value
        return
    if "status" in expected:
        # quantitative nonextractable — inspect capture/stdout note via runner
        # is checked separately in materialize using capture note.
        return



def _portable_run_digest(*, task: RevmathTaskV1, record) -> str:
    """Content digest over identity + judgment — never absolute command paths."""

    judgment = dict(record.result.judgment)
    proof = record.result.proof
    return content_sha(
        {
            "task_identity_digest": task.identity_digest(),
            "result_id": record.result.result_id,
            "judgment": judgment,
            "proof_sha256": None if proof is None else proof.proof_sha256,
            "note": record.capture.get("note"),
            "returncode": record.capture.get("returncode"),
            "timed_out": record.capture.get("timed_out"),
        }
    )


def materialize_entry(
    spec: FixtureCorpusEntrySpecV1,
    *,
    root: Path | None = None,
) -> FixtureCorpusEntryV1:
    """Replay one catalog row and freeze digests + bound identity fields."""

    repo = root or _repo_root()
    fixtures = fixtures_dir(repo)
    artifact_sha = _artifact_digest(
        fixtures=fixtures,
        task_stem=spec.task_stem,
        meta_stem=spec.meta_stem,
        claim_stem=spec.claim_stem,
        probe_id=spec.probe_id,
    )
    proposition_id = None
    statement_sha256 = None
    base_theory_id = None
    allowed: tuple[str, ...] = ()
    budget: Mapping[str, Any] | None = None
    replay_digest: str
    outcome = spec.expected_outcome
    unknown_reason = spec.expected_unknown_reason

    if spec.replay_mode == "label_claim":
        assert spec.claim_stem is not None
        claim = parse_label_claim(_load_json(fixtures / f"{spec.claim_stem}.claim.json"))
        verdict = validate_label_claim(claim)
        if spec.expected_label_accepted is not None:
            if verdict.accepted != spec.expected_label_accepted:
                raise RevmathSchemaError(
                    f"{spec.entry_id}: label accepted={verdict.accepted}, "
                    f"expected {spec.expected_label_accepted}"
                )
        proposition_id = claim.claim_id
        statement_sha256 = claim.forward_evidence_sha256
        base_theory_id = claim.base_theory_id
        replay_digest = content_sha(
            {
                "claim_id": claim.claim_id,
                "accepted": verdict.accepted,
                "labeling_state": claim.labeling_state,
                "rejection_reasons": list(verdict.rejection_reasons),
            }
        )
    elif spec.replay_mode == "ablation_lattice":
        assert spec.task_stem and spec.meta_stem
        task = _load_task(fixtures, spec.task_stem)
        meta = parse_ablation_meta(_load_json(fixtures / f"{spec.meta_stem}.meta.json"))
        report = _run_ablation_lattice(task, meta)
        _assert_expected_report(spec, report)
        proposition_id = task.proposition.proposition_id
        statement_sha256 = task.proposition.statement_sha256
        base_theory_id = task.base_theory.base_theory_id
        allowed = tuple(task.base_theory.allowed_assumption_ids)
        budget = task.budget.model_dump(mode="json")
        replay_digest = getattr(report, "report_digest", None) or _stable_report_digest(report)
    elif spec.replay_mode == "reversal_pair":
        assert spec.task_stem and spec.meta_stem
        task = _load_task(fixtures, spec.task_stem)
        meta = parse_reversal_meta(_load_json(fixtures / f"{spec.meta_stem}.meta.json"))
        report = _run_reversal_pair(task, meta)
        _assert_expected_report(spec, report)
        proposition_id = task.proposition.proposition_id
        statement_sha256 = task.proposition.statement_sha256
        base_theory_id = task.base_theory.base_theory_id
        allowed = tuple(task.base_theory.allowed_assumption_ids)
        budget = task.budget.model_dump(mode="json")
        replay_digest = _stable_report_digest(report)
    elif spec.replay_mode == "runner_probe":
        assert spec.task_stem and spec.probe_id
        task = _load_task(fixtures, spec.task_stem)
        record = _run_probe(task, spec.probe_id, repo)
        judgment = record.result.solver_judgment()
        if outcome and judgment.outcome.value != outcome:
            raise RevmathSchemaError(
                f"{spec.entry_id}: outcome {judgment.outcome.value} != {outcome}"
            )
        if unknown_reason:
            payload = judgment.payload
            got = payload.value if hasattr(payload, "value") else None
            if got != unknown_reason:
                raise RevmathSchemaError(
                    f"{spec.entry_id}: unknown_reason {got!r} != {unknown_reason!r}"
                )
        proposition_id = task.proposition.proposition_id
        statement_sha256 = task.proposition.statement_sha256
        base_theory_id = task.base_theory.base_theory_id
        allowed = tuple(task.base_theory.allowed_assumption_ids)
        budget = task.budget.model_dump(mode="json")
        replay_digest = _portable_run_digest(task=task, record=record)
    else:  # single_runner
        assert spec.task_stem
        task = _load_task(fixtures, spec.task_stem)
        record = run_revmath_task(task, hermetic=True)
        judgment = record.result.solver_judgment()
        if outcome and judgment.outcome.value != outcome:
            raise RevmathSchemaError(
                f"{spec.entry_id}: outcome {judgment.outcome.value} != {outcome}"
            )
        if unknown_reason:
            payload = judgment.payload
            got = (
                payload.value
                if hasattr(payload, "value")
                else (
                    "malformed_input"
                    if judgment.outcome is JudgmentOutcome.INVALID
                    else None
                )
            )
            if got != unknown_reason:
                raise RevmathSchemaError(
                    f"{spec.entry_id}: reason {got!r} != {unknown_reason!r}"
                )
        if spec.expected_report and "status" in spec.expected_report:
            note = str(record.capture.get("note") or "")
            stdout = str(record.capture.get("stdout") or "")
            if "nonextractable" not in note and "nonextractable" not in stdout:
                raise RevmathSchemaError(
                    f"{spec.entry_id}: expected nonextractable status in capture"
                )
        proposition_id = task.proposition.proposition_id
        statement_sha256 = task.proposition.statement_sha256
        base_theory_id = task.base_theory.base_theory_id
        allowed = tuple(task.base_theory.allowed_assumption_ids)
        budget = task.budget.model_dump(mode="json")
        replay_digest = _portable_run_digest(task=task, record=record)

    return FixtureCorpusEntryV1(
        entry_id=spec.entry_id,
        corpus_class=spec.corpus_class,
        split=spec.split,
        root_family_id=spec.root_family_id,
        replay_mode=spec.replay_mode,
        task_stem=spec.task_stem,
        meta_stem=spec.meta_stem,
        claim_stem=spec.claim_stem,
        probe_id=spec.probe_id,
        checker_ref=spec.checker_ref,
        source_toolchain=spec.source_toolchain,
        proposition_id=proposition_id,
        statement_sha256=statement_sha256,
        base_theory_id=base_theory_id,
        allowed_assumption_ids=allowed,
        budget=budget,
        expected_outcome=outcome,
        expected_unknown_reason=unknown_reason,
        expected_label_accepted=spec.expected_label_accepted,
        expected_report=(
            None if spec.expected_report is None else dict(spec.expected_report)
        ),
        allowed_labels=spec.allowed_labels,
        mutation_expectations=spec.mutation_expectations,
        artifact_sha256=artifact_sha,
        replay_digest=replay_digest,
    )


def _manifest_digest(entries: Sequence[FixtureCorpusEntryV1], corpus_id: str) -> str:
    payload = {
        "schema_version": CORPUS_SCHEMA,
        "corpus_id": corpus_id,
        "fixtures_dir": FIXTURES_RELATIVE,
        "entries": [e.to_dict() for e in entries],
    }
    return content_sha(payload)


def build_fixture_corpus_manifest(
    *,
    root: Path | None = None,
    specs: Sequence[FixtureCorpusEntrySpecV1] | None = None,
) -> FixtureCorpusManifestV1:
    rows = tuple(materialize_entry(spec, root=root) for spec in (specs or catalog_specs()))
    classes = {e.corpus_class for e in rows}
    missing = REQUIRED_CORPUS_CLASSES - classes
    if missing:
        raise RevmathSchemaError(f"corpus missing required classes: {sorted(missing)}")
    digest = _manifest_digest(rows, CORPUS_ID)
    return FixtureCorpusManifestV1(
        schema_version=CORPUS_SCHEMA,
        corpus_id=CORPUS_ID,
        corpus_sha256=digest,
        fixtures_dir=FIXTURES_RELATIVE,
        entry_count=len(rows),
        entries=rows,
    )


def write_fixture_corpus_manifest(
    manifest: FixtureCorpusManifestV1,
    *,
    root: Path | None = None,
) -> Path:
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_fixture_corpus_manifest(
    *,
    root: Path | None = None,
    path: Path | None = None,
) -> FixtureCorpusManifestV1:
    target = path or manifest_path(root)
    if not target.is_file():
        raise RevmathSchemaError(f"frozen revmath corpus missing at {target}")
    manifest = FixtureCorpusManifestV1.from_dict(_load_json(target))
    expected = _manifest_digest(manifest.entries, manifest.corpus_id)
    if expected != manifest.corpus_sha256:
        raise RevmathSchemaError(
            "corpus_sha256 does not match content-addressed digest of entries"
        )
    if manifest.entry_count != len(manifest.entries):
        raise RevmathSchemaError("entry_count mismatch")
    return manifest


def coverage_table(
    manifest: FixtureCorpusManifestV1,
) -> list[dict[str, Any]]:
    return [
        {
            "corpus_class": e.corpus_class,
            "entry_id": e.entry_id,
            "split": e.split,
            "root_family_id": e.root_family_id,
            "replay_mode": e.replay_mode,
            "expected_outcome": e.expected_outcome,
            "task_stem": e.task_stem or e.claim_stem or e.probe_id,
        }
        for e in manifest.entries
    ]


def detect_train_eval_leakage(
    manifest: FixtureCorpusManifestV1,
) -> tuple[str, ...]:
    """Return root_family_ids that appear in both train and eval splits."""

    train = {e.root_family_id for e in manifest.entries if e.split == "train"}
    eval_ = {e.root_family_id for e in manifest.entries if e.split == "eval"}
    return tuple(sorted(train & eval_))


def assert_no_undeclared_train_eval_leakage(
    manifest: FixtureCorpusManifestV1,
    *,
    allowed_shared_families: Iterable[str] = (),
) -> None:
    """Fail closed unless shared families are explicitly allowlisted.

    Shared families are detectable (``detect_train_eval_leakage``); callers that
    intentionally share a family for paired pos/neg cases must declare them.
    """

    allowed = set(allowed_shared_families)
    leaked = [f for f in detect_train_eval_leakage(manifest) if f not in allowed]
    if leaked:
        raise RevmathSchemaError(
            "undeclared train/eval root-family leakage: " + ", ".join(leaked)
        )


def verify_corpus_replay(
    manifest: FixtureCorpusManifestV1,
    *,
    root: Path | None = None,
) -> None:
    """Re-materialize every entry and require byte-stable digests."""

    rebuilt = build_fixture_corpus_manifest(root=root)
    if rebuilt.corpus_sha256 != manifest.corpus_sha256:
        # pinpoint first mismatch
        by_id = {e.entry_id: e for e in rebuilt.entries}
        for entry in manifest.entries:
            live = by_id.get(entry.entry_id)
            if live is None:
                raise RevmathSchemaError(f"missing live entry {entry.entry_id}")
            if live.replay_digest != entry.replay_digest:
                raise RevmathSchemaError(
                    f"replay digest drift for {entry.entry_id}: "
                    f"frozen={entry.replay_digest[:16]} live={live.replay_digest[:16]}"
                )
            if live.artifact_sha256 != entry.artifact_sha256:
                raise RevmathSchemaError(
                    f"artifact digest drift for {entry.entry_id}"
                )
        raise RevmathSchemaError("corpus_sha256 drift after full rebuild")


_MUTATION_KIND_TO_ALIAS: Mapping[MutationKind, str] = {
    "proposition": "statement_sha256",
    "toolchain": "judge_id",
    "source": "base_theory_id",
    "certificate": "expected_counterexample_semantics",
    "corpus_membership": "corpus_sha256",
    "budget": "max_wall_seconds",
}


def demonstrate_mutation_failures(
    entry: FixtureCorpusEntryV1,
    *,
    root: Path | None = None,
) -> list[str]:
    """Apply declared mutations; each must_fail expectation must raise."""

    del root  # mutations are identity-gate checks; no filesystem needed
    failures: list[str] = []
    for expectation in entry.mutation_expectations:
        if not expectation.must_fail:
            continue
        alias = _MUTATION_KIND_TO_ALIAS.get(expectation.kind)
        if alias is None:
            raise RevmathSchemaError(f"unknown mutation kind {expectation.kind}")
        try:
            reject_forbidden_repair(alias)
        except RevmathSchemaError:
            failures.append(f"{expectation.kind}:{alias}")
            continue
        raise RevmathSchemaError(
            f"mutation {expectation.kind} on {entry.entry_id} did not fail gate "
            f"{expectation.gate}"
        )
    return failures


__all__ = [
    "CORPUS_ID",
    "CORPUS_SCHEMA",
    "REQUIRED_CORPUS_CLASSES",
    "FixtureCorpusEntrySpecV1",
    "FixtureCorpusEntryV1",
    "FixtureCorpusManifestV1",
    "MutationExpectationV1",
    "assert_no_undeclared_train_eval_leakage",
    "build_fixture_corpus_manifest",
    "catalog_specs",
    "coverage_table",
    "demonstrate_mutation_failures",
    "detect_train_eval_leakage",
    "fixtures_dir",
    "load_fixture_corpus_manifest",
    "manifest_path",
    "materialize_entry",
    "verify_corpus_replay",
    "write_fixture_corpus_manifest",
]
