"""SLM-232 (PEL0-01): preference build-pairs eval-holdout leakage gate stress test.

The test-data phase enforces train/eval disjointness mechanically:
``build_test_data`` (``harnesses/test_data/pipeline.py``) loads train
fingerprints via :func:`slm_training.data.leakage.load_train_fingerprints` and
rejects (raises ``ValueError``) any candidate eval record whose id, prompt,
OpenUI text, structural shape, or (prompt, openui) pair already exists in the
training corpus (:func:`slm_training.data.leakage.find_leakage`).

``contracts.md`` states the same disjointness expectation for the *opposite*
direction: "Never train on eval-feedback holdouts." ``preference.md``
restates it: "Never train on eval-feedback holdouts." Neither the preference
CLI (:mod:`scripts.train_preference`) nor its library owner
(:mod:`slm_training.harnesses.preference`) imports
:mod:`slm_training.data.leakage`, or any other leakage/disjointness helper --
``build-pairs`` takes ``--train-records <path>`` and feeds whatever is at that
path straight into :func:`collect_pairs_with_generator` /
:func:`write_pairs`. Nothing checks whether those records are the committed
held-out / adversarial / ood suites verbatim.

This harness asks a narrow, falsifiable, CPU-only question: **does the real,
unmodified preference build-pairs pipeline (``collect_pairs_with_generator`` +
``write_pairs``, exercised exactly as ``scripts/train_preference.py
build-pairs`` calls them with no ``--from-checkpoint``) silently accept and
emit valid preference pairs when its ``--train-records`` input is, byte for
byte, a genuine eval suite freshly built by the real, unmodified
``build_test_data`` pipeline -- with no error, no warning, and no pair-level
provenance flag distinguishing it from ordinary train data?**

No new gate is implemented and no existing gate is changed. This only
exercises the real ``build_test_data`` / ``collect_pairs_with_generator`` /
``write_pairs`` / ``find_leakage`` / ``load_train_fingerprints`` functions
(never a reimplementation), builds a genuine fixture-scale eval bundle in this
run, feeds it through the real preference pair-building path, and compares
the outcome against an *illustrative* (not implemented, not gating) candidate
check that reuses the same fingerprinting module test-data already trusts,
pointed at the eval suite instead of the train manifest.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.data.leakage import find_leakage, load_train_fingerprints
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses import preference as preference_pkg
from slm_training.harnesses.preference import (
    collect_pairs_with_generator,
    write_pairs,
)
from slm_training.harnesses.preference import train as preference_train_module
from slm_training.harnesses.quality import soft_corrupt_openui
from slm_training.harnesses.test_data.pipeline import TestDataConfig, build_test_data
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "LeakageArm",
    "LeakageArmResult",
    "PreferenceEvalLeakageGateReport",
    "build_default_arms",
    "render_markdown",
    "run_eval_leakage_gate_stress_fixture",
]

MATRIX_VERSION = "pel0-01-v1"
MATRIX_SET = "slm232_preference_eval_leakage_gate"
EXPERIMENT_ID = "slm232-preference-eval-leakage-gate"

TRAIN_SEEDS_PATH = Path("src/slm_training/resources/train_seeds.jsonl")
TEST_SEEDS_PATH = Path("src/slm_training/resources/test_seeds.jsonl")
# build_test_data(source="fixture") only ever populates suites present in the
# committed test_seeds.jsonl; rico_held needs a live/cached RICO source and is
# intentionally out of scope (CPU/offline-only fixture run).
FIXTURE_SUITES = ("smoke", "held_out", "adversarial", "ood")

# Modules exercised by `scripts/train_preference.py build-pairs` for the
# default (no --from-checkpoint) path. Source-scanned for any reference to
# the leakage/disjointness module test-data trusts.
_AUDITED_MODULES = {
    "slm_training.harnesses.preference (__init__)": preference_pkg,
    "slm_training.harnesses.preference.train": preference_train_module,
}

_HYPOTHESIS = (
    "The real collect_pairs_with_generator / write_pairs pipeline, exercised "
    "exactly as scripts/train_preference.py build-pairs calls them with no "
    "--from-checkpoint, has no leakage/disjointness enforcement against eval "
    "suites: it accepts records drawn verbatim from a freshly-built, genuine "
    "held_out/adversarial/ood/smoke eval suite (built by the real, unmodified "
    "build_test_data pipeline) and emits valid preference pairs with no "
    "error, no warning, and no eval-provenance flag -- exactly as if those "
    "records were ordinary train data -- while the equivalent train-side "
    "disjointness check (find_leakage / load_train_fingerprints) that "
    "build_test_data already enforces would flag every one of those records "
    "as 100% fingerprint-identical to the eval suite it came from."
)

_FALSIFIER = (
    "collect_pairs_with_generator / write_pairs (or scripts/train_preference.py "
    "build-pairs around them) reject, warn on, or otherwise flag "
    "eval-suite-sourced records differently from ordinary train records; or "
    "the preference package already imports/calls the leakage/disjointness "
    "module."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, preference training "
    "step, or ship-gate claim is made or implied.",
    "This exercises the real, unmodified build_test_data, "
    "collect_pairs_with_generator, write_pairs, find_leakage, and "
    "load_train_fingerprints functions against a tiny (16-record) committed "
    "fixture corpus, not a production-scale evaluation or training run.",
    "The fingerprint-match candidate check (reusing load_train_fingerprints / "
    "find_leakage, pointed at the eval suite instead of a train manifest) is "
    "an illustrative diagnostic only. It is not implemented in the preference "
    "harness or scripts/train_preference.py, not proposed as the correct fix, "
    "and passing/failing it makes no gate or promotion claim.",
    "Whether training a real preference model on eval-suite-identical pairs "
    "actually harms downstream ship-gate scores (as opposed to only being an "
    "unenforced provenance gap) is not measured here; this harness is about "
    "the build-pairs *pipeline*, not preference training dynamics.",
    "The default (no --from-checkpoint) build-pairs candidate generator is "
    "reproduced inline (gold + soft-corrupt reject) to match the CLI exactly; "
    "it is not imported from scripts/train_preference.py because that module "
    "is a __main__ script, not an importable library function.",
    "The static source-audit only inspects the preference package modules "
    "reachable from build-pairs' default path; it does not prove the absence "
    "of leakage checks anywhere else in the repository.",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return _sha256(_canonical_json(value))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_build_pairs_generator(record: ExampleRecord) -> list[str]:
    """Mirror scripts/train_preference.py build-pairs' default candidate
    generator exactly (no --from-checkpoint, --soft-corrupt default on,
    --corrupt not passed): gold candidate plus one soft-corrupted reject."""
    return [record.openui, soft_corrupt_openui(record.openui)]


def _build_fixture_eval_suites(root: Path, tmp_dir: Path) -> dict[str, list[ExampleRecord]]:
    """Run the real, unmodified build_test_data pipeline (source=fixture,
    offline, no RICO/HF download) with the committed train_seeds.jsonl as the
    train manifest, producing a genuine held_out/adversarial/ood/smoke eval
    bundle for this run. Raises if build_test_data itself finds fixture/train
    leakage (it would -- correctly -- refuse to build in that case)."""
    train_manifest_path = tmp_dir / "train_manifest.json"
    train_manifest_path.write_text(
        json.dumps({"records": str(root / TRAIN_SEEDS_PATH)}) + "\n",
        encoding="utf-8",
    )
    config = TestDataConfig(
        seed_path=root / TEST_SEEDS_PATH,
        rico_path=None,
        source="fixture",
        output_root=tmp_dir / "eval",
        version="pel0-01-fixture",
        suites=FIXTURE_SUITES,
        train_manifest=train_manifest_path,
        require_train_manifest=True,
    )
    from slm_training.dsl.schema import load_jsonl

    result = build_test_data(config)
    suite_paths = result["manifest"]["suites"]
    return {
        suite: load_jsonl(path)
        for suite, path in suite_paths.items()
        if suite in FIXTURE_SUITES
    }


def _fingerprints_for_records(records: list[ExampleRecord], tmp_dir: Path, name: str) -> dict[str, set[str]]:
    """Reuse load_train_fingerprints -- the exact function build_test_data
    calls for its train-side disjointness check -- pointed at an arbitrary
    ExampleRecord source. The function is generic (it fingerprints whatever
    JSONL a manifest's "records" key names); applying it to an eval suite
    instead of a train manifest is not a reimplementation."""
    from slm_training.dsl.schema import write_jsonl

    records_path = tmp_dir / f"{name}_records.jsonl"
    write_jsonl(records_path, records)
    manifest_path = tmp_dir / f"{name}_manifest.json"
    manifest_path.write_text(
        json.dumps({"records": str(records_path)}) + "\n", encoding="utf-8"
    )
    return load_train_fingerprints(manifest_path)


def _run_real_build_pairs(
    records: list[ExampleRecord], tmp_dir: Path, name: str
) -> tuple[bool, str | None, int]:
    """Exact mirror of `scripts/train_preference.py build-pairs` with no
    --from-checkpoint: real collect_pairs_with_generator + write_pairs."""
    out_path = tmp_dir / f"{name}_pairs.jsonl"
    try:
        pairs = collect_pairs_with_generator(
            records,
            _default_build_pairs_generator,
            prefer_valid_rejects=True,
            structure_only=True,
            include_gold=True,
            generator_checkpoint=None,
        )
        n = write_pairs(out_path, pairs)
        return True, None, n
    except Exception as exc:  # noqa: BLE001 - report any real failure, don't hide it
        return False, f"{type(exc).__name__}: {exc}", 0


@dataclass(frozen=True)
class LeakageArm:
    """One preference build-pairs input source."""

    name: str
    description: str
    records_source: str
    is_negative_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "records_source": self.records_source,
            "is_negative_control": self.is_negative_control,
        }


def build_default_arms() -> list[LeakageArm]:
    """One negative-control arm (ordinary train records) plus four arms
    feeding a genuine eval suite straight into build-pairs."""
    return [
        LeakageArm(
            name="train_seeds_control",
            description=(
                "Ordinary train records (src/slm_training/resources/"
                "train_seeds.jsonl) fed to build-pairs as intended. Negative "
                "control: should build successfully and NOT fingerprint-match "
                "the held_out eval suite."
            ),
            records_source="train_seeds",
            is_negative_control=True,
        ),
        LeakageArm(
            name="held_out_as_train_records",
            description=(
                "The real, freshly-built held_out eval suite fed to build-pairs "
                "as --train-records."
            ),
            records_source="held_out",
        ),
        LeakageArm(
            name="adversarial_as_train_records",
            description=(
                "The real, freshly-built adversarial eval suite fed to "
                "build-pairs as --train-records."
            ),
            records_source="adversarial",
        ),
        LeakageArm(
            name="ood_as_train_records",
            description=(
                "The real, freshly-built ood eval suite fed to build-pairs as "
                "--train-records."
            ),
            records_source="ood",
        ),
        LeakageArm(
            name="smoke_as_train_records",
            description=(
                "The real, freshly-built smoke eval suite fed to build-pairs "
                "as --train-records."
            ),
            records_source="smoke",
        ),
    ]


@dataclass(frozen=True)
class LeakageArmResult:
    """The real build-pairs outcome for one arm, plus the illustrative
    fingerprint-match candidate check."""

    arm: LeakageArm
    n_records: int
    build_pairs_succeeded: bool
    build_pairs_error: str | None
    pairs_written: int
    match_rate_against_source_suite_fingerprints: float
    n_fingerprint_reasons_total: int
    gameable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.to_dict(),
            "n_records": self.n_records,
            "build_pairs_succeeded": self.build_pairs_succeeded,
            "build_pairs_error": self.build_pairs_error,
            "pairs_written": self.pairs_written,
            "match_rate_against_source_suite_fingerprints": (
                self.match_rate_against_source_suite_fingerprints
            ),
            "n_fingerprint_reasons_total": self.n_fingerprint_reasons_total,
            "gameable": self.gameable,
        }


def _evaluate_arm(
    arm: LeakageArm,
    suites: dict[str, list[ExampleRecord]],
    suite_fps: dict[str, dict[str, set[str]]],
    tmp_dir: Path,
) -> LeakageArmResult:
    records = suites[arm.records_source]
    succeeded, error, n_written = _run_real_build_pairs(records, tmp_dir, arm.name)

    # Illustrative candidate check: control arm is checked against the
    # held_out suite (cross-corpus, should not match); eval arms are checked
    # against the suite they were literally drawn from (should match fully).
    reference_suite = "held_out" if arm.is_negative_control else arm.records_source
    fps = suite_fps[reference_suite]
    reasons_total = 0
    matched = 0
    for record in records:
        reasons = find_leakage(record, fps)
        if reasons:
            matched += 1
            reasons_total += len(reasons)
    match_rate = matched / len(records) if records else 0.0

    gameable = bool(
        succeeded
        and not arm.is_negative_control
        and match_rate == 1.0
    )

    return LeakageArmResult(
        arm=arm,
        n_records=len(records),
        build_pairs_succeeded=succeeded,
        build_pairs_error=error,
        pairs_written=n_written,
        match_rate_against_source_suite_fingerprints=match_rate,
        n_fingerprint_reasons_total=reasons_total,
        gameable=gameable,
    )


def _static_leakage_import_audit() -> dict[str, bool]:
    """Source-scan the preference modules build-pairs actually calls for any
    reference to the leakage/disjointness module or its symbols."""
    needles = ("data.leakage", "find_leakage", "load_train_fingerprints")
    audit: dict[str, bool] = {}
    for label, module in _AUDITED_MODULES.items():
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError):
            source = ""
        audit[label] = any(needle in source for needle in needles)
    return audit


def _resolve_disposition(
    results: list[LeakageArmResult],
    static_audit: dict[str, bool],
) -> tuple[str, str]:
    control = next((r for r in results if r.arm.is_negative_control), None)
    controls_ok = bool(
        control
        and control.build_pairs_succeeded
        and control.match_rate_against_source_suite_fingerprints < 0.5
    )
    if not controls_ok:
        return (
            "inconclusive",
            "The negative-control arm (ordinary train records) either failed "
            "to build pairs or unexpectedly fingerprint-matched the held_out "
            "suite; the fixture does not isolate the eval-leakage question "
            "cleanly.",
        )

    any_module_wired = any(static_audit.values())
    gameable = [r for r in results if r.gameable]
    non_control_count = sum(1 for r in results if not r.arm.is_negative_control)

    if any_module_wired:
        return (
            "no_gap_found",
            "A static source audit found a reference to the leakage/"
            "disjointness module inside the preference build-pairs code "
            "path; the hypothesized gap does not hold as stated.",
        )
    if len(gameable) == non_control_count and gameable:
        names = ", ".join(r.arm.name for r in gameable)
        return (
            "gap_confirmed",
            f"{len(gameable)}/{non_control_count} eval-suite arms ({names}) "
            "were accepted by the real collect_pairs_with_generator / "
            "write_pairs pipeline and produced valid preference pairs while "
            "100% fingerprint-identical to the eval suite they were drawn "
            "from, and a static audit confirms the preference build-pairs "
            "code path never references the leakage/disjointness module "
            "test-data already trusts for the opposite direction. The "
            "documented 'never train on eval-feedback holdouts' invariant "
            "has no enforcement in build-pairs.",
        )
    return (
        "no_gap_found",
        "Not every eval-suite arm was silently accepted at full fingerprint "
        "overlap; the hypothesized gap does not hold as stated.",
    )


@dataclass(frozen=True)
class PreferenceEvalLeakageGateReport:
    """Full fixture report for SLM-232."""

    schema: str = "PreferenceEvalLeakageGateReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm232-preference-eval-leakage-gate"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    fixture_suites: tuple[str, ...] = FIXTURE_SUITES
    results: tuple[LeakageArmResult, ...] = field(default_factory=tuple)
    static_leakage_import_audit: dict[str, bool] = field(default_factory=dict)
    gate_hash: str = ""
    disposition: str = "inconclusive"
    disposition_rationale: str = ""
    honest_caveats: tuple[str, ...] = _HONEST_CAVEATS
    version_stamp: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "matrix_set": self.matrix_set,
            "matrix_version": self.matrix_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "status": self.status,
            "claim_class": self.claim_class,
            "hypothesis": self.hypothesis,
            "falsifier": self.falsifier,
            "fixture_suites": list(self.fixture_suites),
            "results": [r.to_dict() for r in self.results],
            "static_leakage_import_audit": dict(self.static_leakage_import_audit),
            "gate_hash": self.gate_hash,
            "disposition": self.disposition,
            "disposition_rationale": self.disposition_rationale,
            "honest_caveats": list(self.honest_caveats),
            "version_stamp": dict(self.version_stamp),
            "timestamp": self.timestamp,
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreferenceEvalLeakageGateReport":
        results = tuple(
            LeakageArmResult(
                arm=LeakageArm(
                    name=str(r["arm"]["name"]),
                    description=str(r["arm"]["description"]),
                    records_source=str(r["arm"]["records_source"]),
                    is_negative_control=bool(r["arm"].get("is_negative_control", False)),
                ),
                n_records=int(r["n_records"]),
                build_pairs_succeeded=bool(r["build_pairs_succeeded"]),
                build_pairs_error=r.get("build_pairs_error"),
                pairs_written=int(r["pairs_written"]),
                match_rate_against_source_suite_fingerprints=float(
                    r["match_rate_against_source_suite_fingerprints"]
                ),
                n_fingerprint_reasons_total=int(r["n_fingerprint_reasons_total"]),
                gameable=bool(r["gameable"]),
            )
            for r in data.get("results", ())
        )
        return cls(
            schema=str(data.get("schema", "PreferenceEvalLeakageGateReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            fixture_suites=tuple(data.get("fixture_suites", FIXTURE_SUITES)),
            results=results,
            static_leakage_import_audit=dict(data.get("static_leakage_import_audit", {})),
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_eval_leakage_gate_stress_fixture(
    *,
    arms: list[LeakageArm] | None = None,
    run_id: str | None = None,
) -> PreferenceEvalLeakageGateReport:
    """Run every arm through the real build_test_data / collect_pairs_with_generator
    / write_pairs / find_leakage pipeline and compare against the illustrative
    fingerprint-match candidate check."""
    arms = arms if arms is not None else build_default_arms()
    root = _repo_root()

    with tempfile.TemporaryDirectory(prefix="slm232-pel0-01-") as tmp:
        tmp_dir = Path(tmp)
        suites = _build_fixture_eval_suites(root, tmp_dir)
        from slm_training.dsl.schema import load_jsonl

        suites["train_seeds"] = load_jsonl(root / TRAIN_SEEDS_PATH)

        suite_fps = {
            name: _fingerprints_for_records(records, tmp_dir, name)
            for name, records in suites.items()
        }

        results = [_evaluate_arm(arm, suites, suite_fps, tmp_dir) for arm in arms]

    static_audit = _static_leakage_import_audit()
    disposition, rationale = _resolve_disposition(results, static_audit)

    payload = {
        "row_digests": sorted(_digest(r.to_dict()) for r in results),
        "static_audit": static_audit,
    }
    gate_hash = _sha256(_canonical_json(payload))

    return PreferenceEvalLeakageGateReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        results=tuple(results),
        static_leakage_import_audit=static_audit,
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm232_preference_eval_leakage_gate",
        ),
    )


def render_markdown(report: PreferenceEvalLeakageGateReport) -> str:
    lines = [
        f"# SLM-232 (PEL0-01): preference build-pairs eval-holdout leakage gate stress test ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Fixture suites:** {', '.join(report.fixture_suites)}",
        f"**Gate hash:** `{report.gate_hash[:16]}...`",
        f"**Disposition:** {report.disposition} — {report.disposition_rationale}",
        "",
        "## Hypothesis",
        "",
        report.hypothesis,
        "",
        "## Falsifier",
        "",
        report.falsifier,
        "",
        "## Static leakage-import audit",
        "",
        "| module | references leakage module |",
        "| --- | --- |",
    ]
    for label, referenced in sorted(report.static_leakage_import_audit.items()):
        lines.append(f"| {label} | {referenced} |")
    lines += [
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Per-arm results",
        "",
        "| arm | n records | build-pairs succeeded | pairs written | match rate vs. source-suite fingerprints | gameable | control |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        lines.append(
            f"| {r.arm.name} | {r.n_records} | {r.build_pairs_succeeded} | "
            f"{r.pairs_written} | {r.match_rate_against_source_suite_fingerprints:.2f} | "
            f"{r.gameable} | {r.arm.is_negative_control} |"
        )
    lines += [
        "",
        "## Arm descriptions",
        "",
    ]
    for r in report.results:
        lines.append(f"- **{r.arm.name}**: {r.arm.description}")
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`collect_pairs_with_generator`, `write_pairs`, `build_test_data`, or "
        "`find_leakage`, does not train a preference model, and makes no ship "
        "or gate claim. It documents a concrete gap between the documented "
        "'never train on eval-feedback holdouts' invariant and the actual "
        "preference build-pairs code path, as a candidate for a future, "
        "separately reviewed hardening change (never implemented here).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm232_preference_eval_leakage_gate --mode plan-only",
        "python -m scripts.run_slm232_preference_eval_leakage_gate --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
