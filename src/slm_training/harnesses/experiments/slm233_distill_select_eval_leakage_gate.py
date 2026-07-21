"""SLM-233 (DTL0-01): distillation trace-selection eval-holdout leakage gate
stress test.

``collect_trajectories.py`` (P1) defaults to ``--suite held_out``: "Runs
MaskGIT generation over a suite ... records every intermediate canvas / commit
/ remask event ... This store is the substrate for offline self-distillation
SFT." Every trace it appends carries ``meta.source_suite`` set to whichever
suite was rolled out (``held_out`` by default when ``--test-dir`` is used).

``self_distill.py select`` (P2) then calls
:func:`slm_training.harnesses.distill.select.select_traces` /
``filter_traces`` over that same trace store to build the stratified corpus
that ``self_distill.py train`` SFTs on. ``contracts.md`` / ``distill.md``
state the invariant for this exact pair of stages: "Selection data stays
disjoint from frozen evals; never train on held-out benchmark traces."

Neither :func:`slm_training.harnesses.distill.select.filter_traces` nor
:func:`slm_training.harnesses.distill.select.stratum_key` reads
``trace["meta"]["source_suite"]`` (or any other eval-provenance field) at
all -- selection filters only on ``corpus`` label, ``accepted``,
``exact_gold``, policy checkpoint SHA, and decode-config hash. A trace that
was produced by rolling the checkpoint out over the committed ``held_out`` /
``adversarial`` / ``ood`` / ``smoke`` eval suites (the exact CLI default for
P1 collection) is therefore selected into the self-distillation training
corpus exactly like a trace rolled out over ordinary train prompts, with no
error, no warning, and no different treatment.

This harness asks a narrow, falsifiable, CPU-only question: **does the real,
unmodified P2 selection path (``select_traces`` / ``filter_traces``,
exercised exactly as ``scripts/self_distill.py select`` calls them with
default flags) select traces sourced from a frozen eval suite at the same
rate as ordinary train-sourced traces, with no source-suite discrimination?**

No new gate is implemented and no existing gate is changed. This only
exercises the real ``build_test_data`` / ``DecodeTraceRecorder.finalize`` /
``TraceStore.append`` / ``select_traces`` functions (never a
reimplementation) against a genuine fixture-scale eval bundle built in this
run, and a static audit of the P2 selection module's source for any
reference to eval-suite provenance.
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

from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.harnesses import distill as distill_pkg
from slm_training.harnesses.distill import select as distill_select_module
from slm_training.harnesses.distill.select import SelectConfig, select_traces
from slm_training.harnesses.distill.trace_store import DecodeTraceRecorder, TraceStore
from slm_training.harnesses.preference import grammar_score
from slm_training.harnesses.test_data.pipeline import TestDataConfig, build_test_data
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "LeakageArm",
    "LeakageArmResult",
    "DistillSelectEvalLeakageGateReport",
    "build_default_arms",
    "render_markdown",
    "run_eval_leakage_gate_stress_fixture",
]

MATRIX_VERSION = "dtl0-01-v1"
MATRIX_SET = "slm233_distill_select_eval_leakage_gate"
EXPERIMENT_ID = "slm233-distill-select-eval-leakage-gate"

TRAIN_SEEDS_PATH = Path("src/slm_training/resources/train_seeds.jsonl")
TEST_SEEDS_PATH = Path("src/slm_training/resources/test_seeds.jsonl")
# build_test_data(source="fixture") only ever populates suites present in the
# committed test_seeds.jsonl; rico_held needs a live/cached RICO source and is
# intentionally out of scope (CPU/offline-only fixture run).
FIXTURE_SUITES = ("smoke", "held_out", "adversarial", "ood")

# Modules exercised by `scripts/self_distill.py select` for the default P2
# selection path. Source-scanned for any reference to eval-suite provenance.
_AUDITED_MODULES = {
    "slm_training.harnesses.distill.select": distill_select_module,
    "slm_training.harnesses.distill (__init__)": distill_pkg,
}
# NOTE: deliberately excludes the bare string "leakage" -- select.py imports
# slm_training.data.leakage.fingerprint_openui_structure for stratum-key
# clustering (grouping traces by structural shape), not for eval-suite
# disjointness, so that substring alone is not evidence the gap is enforced.
_NEEDLES = ("source_suite", "held_out", "eval_suite", "test_dir")

_HYPOTHESIS = (
    "The real select_traces / filter_traces pipeline, exercised exactly as "
    "scripts/self_distill.py select calls them with default flags, has no "
    "discrimination against traces whose meta.source_suite marks them as "
    "coming from a frozen eval suite: it selects traces rolled out over the "
    "committed held_out/adversarial/ood/smoke eval suites (the CLI default "
    "for P1 trace collection) into the self-distillation training corpus at "
    "the same rate as traces rolled out over ordinary train prompts, with no "
    "error, no warning, and no different treatment."
)

_FALSIFIER = (
    "select_traces / filter_traces (or scripts/self_distill.py select around "
    "them) reject, warn on, drop, or otherwise select eval-suite-sourced "
    "traces at a different rate than ordinary train-sourced traces; or the "
    "distill.select module already reads meta.source_suite (or any other "
    "eval-provenance field)."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, self-distillation "
    "SFT step, or ship-gate claim is made or implied.",
    "This exercises the real, unmodified build_test_data, "
    "DecodeTraceRecorder.finalize, TraceStore.append, and select_traces "
    "functions against a tiny (16-record) committed fixture corpus, not a "
    "production-scale evaluation or training run.",
    "Traces are constructed directly via the real DecodeTraceRecorder / "
    "TraceStore API rather than by running an actual MaskGIT rollout through "
    "a checkpoint (that would need a real trained model and would not "
    "change what select_traces sees: meta.source_suite, labels, and reward "
    "are the same fields collect_trajectories.py writes). The synthesized "
    "final text is a minimally whitespace-mutated copy of each record's gold "
    "OpenUI (grammar-valid, not exact_gold) standing in for a plausible "
    "accepted decode; it is not a claim about real model output quality.",
    "Whether training a real self-distillation SFT step on eval-suite-"
    "sourced traces actually harms downstream ship-gate scores (as opposed "
    "to only being an unenforced provenance gap) is not measured here; this "
    "harness is about the P2 selection *pipeline*, not distillation training "
    "dynamics.",
    "The static source-audit only inspects the distill.select and distill "
    "package modules reachable from self_distill.py select's default path; "
    "it does not prove the absence of leakage checks anywhere else in the "
    "repository, and does not inspect P1 collect_trajectories.py itself "
    "(which is the CLI that writes meta.source_suite in the first place, not "
    "the selection stage under test).",
    "distill.select does import slm_training.data.leakage."
    "fingerprint_openui_structure, but only to build the structural-shape "
    "component of a trace's stratification key (coverage-over-score "
    "sampling), never to compare against an eval-suite fingerprint set; the "
    "static audit's needle list deliberately excludes the bare substring "
    "'leakage' for this reason and checks for source_suite/held_out/"
    "eval_suite/test_dir instead.",
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
        version="dtl0-01-fixture",
        suites=FIXTURE_SUITES,
        train_manifest=train_manifest_path,
        require_train_manifest=True,
    )
    result = build_test_data(config)
    suite_paths = result["manifest"]["suites"]
    return {
        suite: load_jsonl(path)
        for suite, path in suite_paths.items()
        if suite in FIXTURE_SUITES
    }


def _mutate_valid(openui: str) -> str:
    """Grammar-preserving, non-exact-match mutation standing in for a
    plausible accepted decode (see honest caveats: this is not a real model
    rollout). Doubles the whitespace around the first top-level '=' so the
    parser still accepts the program but the raw text no longer strip()-
    matches the gold, keeping labels.exact_gold False without changing
    grammar_score."""
    if " = " in openui:
        return openui.replace(" = ", "  = ", 1)
    return openui + " "


def _build_trace(
    record: ExampleRecord,
    *,
    source_suite: str | None,
) -> dict[str, Any]:
    """Build one decode trace via the real DecodeTraceRecorder API -- the
    same finalize() call collect_trajectories.py makes per rollout -- with a
    synthesized (not model-generated; see honest caveats) accepted final
    text and the same meta fields collect_trajectories.py writes, including
    source_suite exactly as it would be set for a --test-dir/--suite run
    (None when --records is used instead)."""
    recorder = DecodeTraceRecorder(record_canvases=False, record_support=False)
    recorder.begin(decode_policy="maskgit")
    text = _mutate_valid(record.openui)
    recorder.end(canvas=None, text=text)
    g = grammar_score(text)
    reward = {"grammar": g}
    labels = {
        "accepted": g > 0.0,
        "exact_gold": text.strip() == record.openui.strip(),
    }
    return recorder.finalize(
        final_text=text,
        reward=reward,
        labels=labels,
        record_id=record.id,
        prompt=record.prompt,
        sample_index=0,
        source_suite=source_suite,
        policy_checkpoint_sha=None,
        policy_checkpoint="dtl0-01-fixture",
        decode_config_hash="dtl0-01-fixture-hash",
        tokenizer_version=None,
        tokenizer_sha=None,
        context_text=None,
        seed=0,
    )


@dataclass(frozen=True)
class LeakageArm:
    """One P1 collection source (a would-be `--suite` value, or None for a
    plain `--records` train-prompt rollout)."""

    name: str
    description: str
    records_source: str
    source_suite: str | None
    is_negative_control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "records_source": self.records_source,
            "source_suite": self.source_suite,
            "is_negative_control": self.is_negative_control,
        }


def build_default_arms() -> list[LeakageArm]:
    """One negative-control arm (traces from ordinary train prompts, no
    --test-dir, so source_suite=None) plus four arms whose traces carry
    meta.source_suite set to a genuine frozen eval suite -- exactly what
    `collect_trajectories.py --test-dir <dir> --suite <name>` would write."""
    return [
        LeakageArm(
            name="train_prompts_control",
            description=(
                "Traces rolled out over ordinary train prompts (src/"
                "slm_training/resources/train_seeds.jsonl), as if collected "
                "via `collect_trajectories.py --records train_seeds.jsonl` "
                "(no --test-dir, so source_suite=None). Negative control: "
                "should select into the corpus normally."
            ),
            records_source="train_seeds",
            source_suite=None,
            is_negative_control=True,
        ),
        LeakageArm(
            name="held_out_suite_traces",
            description=(
                "Traces rolled out over the real, freshly-built held_out eval "
                "suite, as if collected via `collect_trajectories.py "
                "--test-dir <dir> --suite held_out` (the CLI default)."
            ),
            records_source="held_out",
            source_suite="held_out",
        ),
        LeakageArm(
            name="adversarial_suite_traces",
            description=(
                "Traces rolled out over the real, freshly-built adversarial "
                "eval suite, as if collected via `collect_trajectories.py "
                "--test-dir <dir> --suite adversarial`."
            ),
            records_source="adversarial",
            source_suite="adversarial",
        ),
        LeakageArm(
            name="ood_suite_traces",
            description=(
                "Traces rolled out over the real, freshly-built ood eval "
                "suite, as if collected via `collect_trajectories.py "
                "--test-dir <dir> --suite ood`."
            ),
            records_source="ood",
            source_suite="ood",
        ),
        LeakageArm(
            name="smoke_suite_traces",
            description=(
                "Traces rolled out over the real, freshly-built smoke eval "
                "suite, as if collected via `collect_trajectories.py "
                "--test-dir <dir> --suite smoke`."
            ),
            records_source="smoke",
            source_suite="smoke",
        ),
    ]


@dataclass(frozen=True)
class LeakageArmResult:
    """The real select_traces outcome for one arm's traces, mixed into a
    combined trace store alongside every other arm (mirroring one shared
    trace store that P1 has appended to from multiple collection runs)."""

    arm: LeakageArm
    n_traces: int
    n_selected: int
    selection_rate: float
    gameable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.to_dict(),
            "n_traces": self.n_traces,
            "n_selected": self.n_selected,
            "selection_rate": self.selection_rate,
            "gameable": self.gameable,
        }


def _static_source_suite_audit() -> dict[str, bool]:
    """Source-scan the distill selection modules `self_distill.py select`
    actually calls for any reference to eval-suite provenance fields."""
    audit: dict[str, bool] = {}
    for label, module in _AUDITED_MODULES.items():
        try:
            source = inspect.getsource(module)
        except (OSError, TypeError):
            source = ""
        audit[label] = any(needle in source for needle in _NEEDLES)
    return audit


def _resolve_disposition(
    results: list[LeakageArmResult],
    control_rate: float,
    static_audit: dict[str, bool],
) -> tuple[str, str]:
    controls_ok = control_rate > 0.0
    if not controls_ok:
        return (
            "inconclusive",
            "The negative-control arm (ordinary train-prompt traces) was not "
            "selected at all; the fixture does not isolate the eval-leakage "
            "question cleanly.",
        )

    any_module_wired = any(static_audit.values())
    eval_results = [r for r in results if not r.arm.is_negative_control]
    gameable = [r for r in eval_results if r.gameable]

    if any_module_wired:
        return (
            "no_gap_found",
            "A static source audit found a reference to eval-suite "
            "provenance (source_suite / held_out / test_dir / leakage) "
            "inside the distill P2 selection code path; the hypothesized "
            "gap does not hold as stated.",
        )
    if len(gameable) == len(eval_results) and gameable:
        names = ", ".join(r.arm.name for r in gameable)
        return (
            "gap_confirmed",
            f"{len(gameable)}/{len(eval_results)} eval-suite arms ({names}) "
            "were selected by the real select_traces / filter_traces "
            "pipeline at the same rate as the train-prompt control "
            f"({control_rate:.2f}), and a static audit confirms the P2 "
            "selection code path never reads meta.source_suite or any "
            "other eval-provenance field. The documented 'selection data "
            "stays disjoint from frozen evals; never train on held-out "
            "benchmark traces' invariant has no enforcement in select_traces.",
        )
    return (
        "no_gap_found",
        "Not every eval-suite arm was selected at the same rate as the "
        "train-prompt control; the hypothesized gap does not hold as "
        "stated.",
    )


@dataclass(frozen=True)
class DistillSelectEvalLeakageGateReport:
    """Full fixture report for SLM-233."""

    schema: str = "DistillSelectEvalLeakageGateReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm233-distill-select-eval-leakage-gate"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    fixture_suites: tuple[str, ...] = FIXTURE_SUITES
    results: tuple[LeakageArmResult, ...] = field(default_factory=tuple)
    control_selection_rate: float = 0.0
    static_source_suite_audit: dict[str, bool] = field(default_factory=dict)
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
            "control_selection_rate": self.control_selection_rate,
            "static_source_suite_audit": dict(self.static_source_suite_audit),
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
    def from_dict(cls, data: dict[str, Any]) -> "DistillSelectEvalLeakageGateReport":
        results = tuple(
            LeakageArmResult(
                arm=LeakageArm(
                    name=str(r["arm"]["name"]),
                    description=str(r["arm"]["description"]),
                    records_source=str(r["arm"]["records_source"]),
                    source_suite=r["arm"].get("source_suite"),
                    is_negative_control=bool(r["arm"].get("is_negative_control", False)),
                ),
                n_traces=int(r["n_traces"]),
                n_selected=int(r["n_selected"]),
                selection_rate=float(r["selection_rate"]),
                gameable=bool(r["gameable"]),
            )
            for r in data.get("results", ())
        )
        return cls(
            schema=str(data.get("schema", "DistillSelectEvalLeakageGateReportV1")),
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
            control_selection_rate=float(data.get("control_selection_rate", 0.0)),
            static_source_suite_audit=dict(data.get("static_source_suite_audit", {})),
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
) -> DistillSelectEvalLeakageGateReport:
    """Build one combined trace store (mirroring multiple P1 collection runs
    appending into the same store), select from it via the real
    select_traces / filter_traces (exactly as `self_distill.py select` does
    with default flags), and check whether eval-suite-sourced traces are
    selected at the same rate as train-prompt traces."""
    arms = arms if arms is not None else build_default_arms()
    root = _repo_root()

    with tempfile.TemporaryDirectory(prefix="slm233-dtl0-01-") as tmp:
        tmp_dir = Path(tmp)
        suites = _build_fixture_eval_suites(root, tmp_dir)
        suites["train_seeds"] = load_jsonl(root / TRAIN_SEEDS_PATH)

        store = TraceStore(tmp_dir / "traces", run_id=run_id or EXPERIMENT_ID)
        arm_trajectory_ids: dict[str, list[str]] = {}
        for arm in arms:
            records = suites[arm.records_source]
            traj_ids: list[str] = []
            for record in records:
                trace = _build_trace(record, source_suite=arm.source_suite)
                traj_ids.append(store.append(trace))
            arm_trajectory_ids[arm.name] = traj_ids

        # Real, unmodified P2 selection -- exactly `self_distill.py select`'s
        # default invocation (only --budget/--corpus/--seed are CLI flags;
        # per_stratum/require_accepted/exclude_exact_gold keep their
        # SelectConfig defaults, which is what the CLI always runs).
        config = SelectConfig(budget=500, corpus="self_distilled_success", seed=0)
        selected = select_traces(store.iter_traces(), config=config)
        selected_ids = {row.get("trajectory_id") for row in selected}

    results: list[LeakageArmResult] = []
    for arm in arms:
        traj_ids = arm_trajectory_ids[arm.name]
        n_selected = sum(1 for tid in traj_ids if tid in selected_ids)
        rate = n_selected / len(traj_ids) if traj_ids else 0.0
        results.append(
            LeakageArmResult(
                arm=arm,
                n_traces=len(traj_ids),
                n_selected=n_selected,
                selection_rate=rate,
                gameable=False,  # filled in below once control_rate is known
            )
        )

    control = next((r for r in results if r.arm.is_negative_control), None)
    control_rate = control.selection_rate if control else 0.0

    # A non-control arm is "gameable" iff its selection rate matches the
    # control's (within a tiny fixture-scale tolerance for stratum-budget
    # rounding) -- i.e. selection did not discriminate on source_suite.
    def _is_gameable(rate: float) -> bool:
        return abs(rate - control_rate) < 1e-9 and rate > 0.0

    results = [
        LeakageArmResult(
            arm=r.arm,
            n_traces=r.n_traces,
            n_selected=r.n_selected,
            selection_rate=r.selection_rate,
            gameable=(not r.arm.is_negative_control) and _is_gameable(r.selection_rate),
        )
        for r in results
    ]

    static_audit = _static_source_suite_audit()
    disposition, rationale = _resolve_disposition(results, control_rate, static_audit)

    payload = {
        "row_digests": sorted(_digest(r.to_dict()) for r in results),
        "static_audit": static_audit,
        "control_rate": control_rate,
    }
    gate_hash = _sha256(_canonical_json(payload))

    return DistillSelectEvalLeakageGateReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        results=tuple(results),
        control_selection_rate=control_rate,
        static_source_suite_audit=static_audit,
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm233_distill_select_eval_leakage_gate",
        ),
    )


def render_markdown(report: DistillSelectEvalLeakageGateReport) -> str:
    lines = [
        f"# SLM-233 (DTL0-01): distillation trace-selection eval-holdout leakage gate stress test ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
        f"**Fixture suites:** {', '.join(report.fixture_suites)}",
        f"**Gate hash:** `{report.gate_hash[:16]}...`",
        f"**Control selection rate:** {report.control_selection_rate:.2f}",
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
        "## Static source-suite-provenance audit",
        "",
        "| module | references eval-suite provenance |",
        "| --- | --- |",
    ]
    for label, referenced in sorted(report.static_source_suite_audit.items()):
        lines.append(f"| {label} | {referenced} |")
    lines += [
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Per-arm results",
        "",
        "| arm | source_suite | n traces | n selected | selection rate | gameable | control |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        lines.append(
            f"| {r.arm.name} | {r.arm.source_suite} | {r.n_traces} | "
            f"{r.n_selected} | {r.selection_rate:.2f} | {r.gameable} | "
            f"{r.arm.is_negative_control} |"
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
        "`select_traces`, `filter_traces`, `collect_trajectories.py`, or "
        "`self_distill.py`, does not run self-distillation SFT, and makes no "
        "ship or gate claim. It documents a concrete gap between the "
        "documented 'selection data stays disjoint from frozen evals; never "
        "train on held-out benchmark traces' invariant and the actual P2 "
        "selection code path, as a candidate for a future, separately "
        "reviewed hardening change (never implemented here).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm233_distill_select_eval_leakage_gate --mode plan-only",
        "python -m scripts.run_slm233_distill_select_eval_leakage_gate --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
