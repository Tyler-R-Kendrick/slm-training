"""SLM-235 (AEP0-01): annotation export pairing-mechanism consistency probe.

``src/slm_training/harnesses/annotations/__init__.py`` contains two different
algorithms that both turn thumbs-up/thumbs-down feedback into
:class:`~slm_training.harnesses.preference.PreferencePair` rows:

1. **Live/incremental**: :func:`maybe_append_preference_pair`, invoked from
   :func:`persist_annotation` every time the FastAPI ``/api/annotate``
   endpoint (via :class:`~slm_training.harnesses.annotations.store.FileAnnotationStore`)
   durably stores one new annotation. For each new record it scans the
   growing ``feedback.jsonl`` for the most recent opposite-rating record for
   the same prompt and, if found and not an exact duplicate, **appends** one
   pair to ``pairs_path``.
2. **Batch/export**: :func:`export_to_preference_pairs`, invoked by
   :func:`export_all` -- the function backing the documented
   ``slm annotations export`` / ``scripts/export_annotations.py export`` CLI
   subcommand. For each prompt it takes only the single most recent
   thumbs-up and the single most recent content-distinct thumbs-down across
   the *entire* history, and writes the result via
   :func:`~slm_training.harnesses.preference.write_pairs`, which
   ``bench.md``/``annotations.md`` never mention **atomically replaces the
   whole target file** rather than appending.

Both paths default to writing the exact same file:
``DEFAULT_HUMAN_PAIRS_PATH`` (``outputs/data/preference/human_pairs.jsonl``)
is the default ``pairs_path`` for ``FileAnnotationStore.__init__`` *and* the
default ``--pairs`` value in ``scripts/export_annotations.py``'s ``export``
subparser. ``annotations.md`` states the contract as "Stable annotation IDs,
atomic appends, attempt provenance preserved" -- it says nothing about a
second, non-append code path that can be run against the same file.

This harness asks a narrow, falsifiable, CPU-only question: **for a prompt
that receives more than one up/down rating flip over its lifetime, does
running the real, unmodified ``export_to_preference_pairs`` (exactly as the
documented ``slm annotations export`` CLI invokes it, no ``--pairs``
override) against the default pairs path silently discard preference pairs
that the real, unmodified live path (``persist_annotation`` /
``maybe_append_preference_pair``, exactly as the FastAPI annotate endpoint
invokes it) had already durably appended there?**

No new gate is implemented and no existing gate, default, or file-write
behavior is changed. This only exercises the real ``persist_annotation``,
``maybe_append_preference_pair``, ``export_to_preference_pairs``, and
``write_pairs`` functions (never a reimplementation) against synthetic
annotation event sequences.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.annotations import (
    DEFAULT_HUMAN_PAIRS_PATH,
    AnnotationRecord,
    export_to_preference_pairs,
    new_annotation_id,
    persist_annotation,
    utc_now_iso,
)
from slm_training.harnesses.annotations.store import FileAnnotationStore
from slm_training.versioning import build_version_stamp

__all__ = [
    "EXPERIMENT_ID",
    "MATRIX_SET",
    "MATRIX_VERSION",
    "FlipScenario",
    "ScenarioResult",
    "AnnotationExportPairingReport",
    "build_default_scenarios",
    "render_markdown",
    "run_pairing_consistency_fixture",
]

MATRIX_VERSION = "aep0-01-v1"
MATRIX_SET = "slm235_annotation_export_pairing_consistency"
EXPERIMENT_ID = "slm235-annotation-export-pairing-consistency"

EXPORT_SCRIPT_PATH = Path("scripts/export_annotations.py")
STORE_MODULE_PATH = Path("src/slm_training/harnesses/annotations/store.py")

_ORGANIC_SEEDS = (11, 17, 23, 29, 31)
_ORGANIC_PROMPTS_PER_SEED = 5

_HYPOTHESIS = (
    "The live/incremental preference-pairing path (persist_annotation -> "
    "maybe_append_preference_pair, exercised exactly as the FastAPI "
    "/api/annotate endpoint invokes it) and the batch/export preference-"
    "pairing path (export_to_preference_pairs, exercised exactly as the "
    "documented `slm annotations export` CLI invokes it with no --pairs "
    "override) implement different algorithms over identical feedback.jsonl "
    "history -- incremental can append many pairs per prompt (one per rating "
    "flip) while batch keeps at most one pair per prompt (most recent up vs "
    "most recent content-distinct down) -- and because write_pairs "
    "atomically REPLACES its target file rather than appending, running the "
    "documented export CLI against the shared default pairs path after any "
    "live annotation activity silently discards every incremental pair for "
    "a prompt except the one the batch algorithm happens to keep, with no "
    "warning, no backup, and no flag to preserve the discarded history."
)

_FALSIFIER = (
    "For every multi-flip scenario, the pairs recorded incrementally are "
    "fully retained (as a set) after running export_to_preference_pairs "
    "against the same pairs_path; or export_to_preference_pairs/write_pairs "
    "appends instead of replacing; or FileAnnotationStore's default "
    "pairs_path and the export CLI's default --pairs path are not, in fact, "
    "the same file."
)

_HONEST_CAVEATS = (
    "Fixture/wiring evidence only: no checkpoint, GPU run, preference "
    "training step, or ship-gate claim is made or implied.",
    "Annotation content is synthetic placeholder-shaped text generated by "
    "this harness, not real playground/human feedback; it exercises the "
    "real persist_annotation / export_to_preference_pairs / write_pairs "
    "functions, not a reimplementation of their logic.",
    "The organic multi-prompt scenarios use a fixed, small set of 5 seeds; "
    "this bounds but does not exhaustively characterize every possible "
    "rating-sequence shape a real user session could produce.",
    "Whether losing intermediate incremental pairs actually harms "
    "downstream preference-model quality (as opposed to only being a "
    "silent, undocumented data-loss mechanism) is not measured here -- this "
    "harness is about the pairing/export *mechanism*, not preference "
    "training dynamics.",
    "This harness does not change persist_annotation, "
    "maybe_append_preference_pair, export_to_preference_pairs, write_pairs, "
    "or any CLI default. It documents a concrete mechanism gap as a "
    "candidate for a future, separately reviewed hardening change (never "
    "implemented here).",
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


@dataclass(frozen=True)
class FlipScenario:
    """One synthetic annotation session: a set of (prompt, rating-sequence)
    pairs to persist live, in order, before running the batch export."""

    name: str
    description: str
    prompts: tuple[tuple[str, tuple[str, ...]], ...]
    is_negative_control: bool = False
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "prompts": [[p, list(r)] for p, r in self.prompts],
            "is_negative_control": self.is_negative_control,
            "seed": self.seed,
        }


def _organic_events(seed: int, n_prompts: int = _ORGANIC_PROMPTS_PER_SEED) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Generate a small, seeded, organic-shaped set of (prompt, ratings)
    sessions: variable length 2-6 events per prompt, each either repeating
    or flipping the previous rating."""
    rng = random.Random(seed)
    prompts: list[tuple[str, tuple[str, ...]]] = []
    for i in range(n_prompts):
        length = rng.randint(2, 6)
        ratings: list[str] = [rng.choice(["up", "down"])]
        for _ in range(length - 1):
            if rng.random() < 0.6:
                ratings.append("down" if ratings[-1] == "up" else "up")
            else:
                ratings.append(ratings[-1])
        prompts.append((f"AEP0-01 organic prompt seed{seed} #{i}", tuple(ratings)))
    return tuple(prompts)


def build_default_scenarios() -> list[FlipScenario]:
    """One negative-control single-flip scenario, two deterministic
    multi-flip scenarios, and five seeded organic multi-prompt scenarios."""
    scenarios = [
        FlipScenario(
            name="single_flip_control",
            description=(
                "One prompt, one down then one up. Negative control: exactly "
                "one pair should exist both incrementally and after export, "
                "with identical content."
            ),
            prompts=(("AEP0-01 control prompt: single flip", ("down", "up")),),
            is_negative_control=True,
        ),
        FlipScenario(
            name="alternating_four_events",
            description=(
                "One prompt, down/up/down/up. Incremental pairing should "
                "append 3 pairs over the session (up1-vs-down1, "
                "up1-vs-down2, up2-vs-down2); batch export should keep only "
                "1 (up2-vs-down2)."
            ),
            prompts=(("AEP0-01 prompt: alternating four events", ("down", "up", "down", "up")),),
        ),
        FlipScenario(
            name="alternating_five_events_up_first",
            description=(
                "One prompt, up/down/up/down/up (starts and ends on up). "
                "Incremental pairing should append 4 pairs over the "
                "session; batch export should keep only 1."
            ),
            prompts=(
                (
                    "AEP0-01 prompt: alternating five events up first",
                    ("up", "down", "up", "down", "up"),
                ),
            ),
        ),
    ]
    for seed in _ORGANIC_SEEDS:
        scenarios.append(
            FlipScenario(
                name=f"organic_multi_prompt_seed{seed}",
                description=(
                    f"5 prompts, seeded (seed={seed}) 2-6 event sessions each "
                    "with a 60% flip / 40% repeat rating transition, "
                    "mimicking an organic multi-prompt playground session."
                ),
                prompts=_organic_events(seed),
                seed=seed,
            )
        )
    return scenarios


@dataclass(frozen=True)
class ScenarioResult:
    """The real live-then-export outcome for one scenario."""

    scenario: FlipScenario
    n_prompts: int
    n_annotations: int
    incremental_pairs_in_file: int
    batch_pairs_written: int
    pairs_retained: int
    pairs_lost: int
    prompts_with_multi_events: int
    prompts_with_loss: int
    divergent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "n_prompts": self.n_prompts,
            "n_annotations": self.n_annotations,
            "incremental_pairs_in_file": self.incremental_pairs_in_file,
            "batch_pairs_written": self.batch_pairs_written,
            "pairs_retained": self.pairs_retained,
            "pairs_lost": self.pairs_lost,
            "prompts_with_multi_events": self.prompts_with_multi_events,
            "prompts_with_loss": self.prompts_with_loss,
            "divergent": self.divergent,
        }


def _pair_key(pair_dict: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(pair_dict.get("prompt", "")).strip(),
        str(pair_dict.get("chosen", "")).strip(),
        str(pair_dict.get("rejected", "")).strip(),
    )


def _read_pair_keys(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        return []
    keys = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            keys.append(_pair_key(json.loads(line)))
    return keys


def _run_scenario(scenario: FlipScenario, tmp_dir: Path) -> ScenarioResult:
    """Persist every event live via the real FileAnnotationStore (exactly the
    FastAPI /api/annotate path), then run the real batch export against the
    SAME default-shaped pairs path, and compare pair sets before/after."""
    run_dir = tmp_dir / scenario.name
    run_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = run_dir / "feedback.jsonl"
    human_train_path = run_dir / "human_train.jsonl"
    pairs_path = run_dir / "pairs.jsonl"

    store = FileAnnotationStore(
        feedback_path=feedback_path,
        human_train_path=human_train_path,
        pairs_path=pairs_path,
    )

    n_annotations = 0
    counter = 0
    for prompt, ratings in scenario.prompts:
        for rating in ratings:
            counter += 1
            record = AnnotationRecord(
                id=new_annotation_id(),
                ts=utc_now_iso(),
                prompt=prompt,
                openui=f'root = TextContent(":aep0.body_{counter}")\n',
                rating=rating,  # type: ignore[arg-type]
                description=None,
                valid=True,
            )
            store.persist(record)
            n_annotations += 1

    incremental_keys = _read_pair_keys(pairs_path)
    incremental_pairs_in_file = len(incremental_keys)

    batch_result = export_to_preference_pairs(feedback_path, pairs_path)
    batch_keys = _read_pair_keys(pairs_path)
    batch_pairs_written = int(batch_result["count"])
    assert batch_pairs_written == len(batch_keys), (
        "export_to_preference_pairs count did not match the written pairs "
        "file -- fixture invariant broken"
    )

    incremental_set = set(incremental_keys)
    batch_set = set(batch_keys)
    pairs_retained = len(incremental_set & batch_set)
    pairs_lost = len(incremental_set - batch_set)

    prompts_with_multi_events = sum(1 for _, ratings in scenario.prompts if len(ratings) > 2)
    lost_prompts = {key[0] for key in (incremental_set - batch_set)}
    prompts_with_loss = len(lost_prompts)

    return ScenarioResult(
        scenario=scenario,
        n_prompts=len(scenario.prompts),
        n_annotations=n_annotations,
        incremental_pairs_in_file=incremental_pairs_in_file,
        batch_pairs_written=batch_pairs_written,
        pairs_retained=pairs_retained,
        pairs_lost=pairs_lost,
        prompts_with_multi_events=prompts_with_multi_events,
        prompts_with_loss=prompts_with_loss,
        divergent=pairs_lost > 0,
    )


def _static_shared_default_path_audit() -> dict[str, Any]:
    """Confirm the two production entry points really do default to the
    same pairs file: FileAnnotationStore.__init__ (live path) and the
    `export` subparser's --pairs default (scripts/export_annotations.py,
    the batch path)."""
    root = _repo_root()
    export_source = (root / EXPORT_SCRIPT_PATH).read_text(encoding="utf-8")
    store_source = (root / STORE_MODULE_PATH).read_text(encoding="utf-8")
    export_references_default = (
        '"--pairs", type=Path, default=DEFAULT_HUMAN_PAIRS_PATH' in export_source
    )
    store_references_default = "pairs_path: Path = DEFAULT_HUMAN_PAIRS_PATH" in store_source
    live_default_path = str(FileAnnotationStore().pairs_path)
    export_default_path = str(DEFAULT_HUMAN_PAIRS_PATH)
    return {
        "export_cli_default_references_shared_constant": export_references_default,
        "live_store_default_references_shared_constant": store_references_default,
        "live_default_pairs_path": live_default_path,
        "export_cli_default_pairs_path": export_default_path,
        "paths_identical": live_default_path == export_default_path,
    }


def _resolve_disposition(
    results: list[ScenarioResult],
    static_audit: dict[str, Any],
) -> tuple[str, str]:
    control = next((r for r in results if r.scenario.is_negative_control), None)
    controls_ok = bool(
        control
        and control.incremental_pairs_in_file == 1
        and control.batch_pairs_written == 1
        and control.pairs_lost == 0
    )
    if not controls_ok:
        return (
            "inconclusive",
            "The single_flip_control scenario did not produce exactly one "
            "matching pair on both the incremental and batch paths; the "
            "fixture does not isolate the pairing-consistency question "
            "cleanly.",
        )

    if not static_audit.get("paths_identical"):
        return (
            "no_gap_found",
            "FileAnnotationStore's default pairs_path and the export CLI's "
            "default --pairs path are not the same file; the shared-path "
            "collision premise does not hold.",
        )

    non_control = [r for r in results if not r.scenario.is_negative_control]
    deterministic = [r for r in non_control if r.scenario.seed is None]
    organic = [r for r in non_control if r.scenario.seed is not None]

    deterministic_all_lossy = bool(deterministic) and all(r.divergent for r in deterministic)
    organic_with_multi_flip = [r for r in organic if r.prompts_with_multi_events > 0]
    organic_lossy = [r for r in organic_with_multi_flip if r.divergent]

    if deterministic_all_lossy and organic_with_multi_flip and len(organic_lossy) == len(organic_with_multi_flip):
        total_lost = sum(r.pairs_lost for r in non_control)
        return (
            "gap_confirmed",
            f"Every deterministic multi-flip scenario ({len(deterministic)}/"
            f"{len(deterministic)}) and every organic seed with at least one "
            f"multi-event prompt ({len(organic_lossy)}/{len(organic_with_multi_flip)}) "
            "lost at least one incrementally-persisted preference pair after "
            "running the real export_to_preference_pairs against the shared "
            f"default pairs path ({total_lost} pairs lost in total across "
            "non-control scenarios), while the single-flip control matched "
            "exactly. The live and batch pairing algorithms are confirmed "
            "inconsistent, and running the documented `slm annotations "
            "export` command with default arguments after live playground "
            "activity silently discards prior incremental preference-pair "
            "history.",
        )
    if not deterministic_all_lossy and not organic_lossy:
        return (
            "no_gap_found",
            "No scenario with more than one rating flip lost any "
            "incrementally-persisted pair after batch export; the "
            "hypothesized mechanism gap does not hold as stated.",
        )
    return (
        "partial_confirmation",
        f"Loss was observed in {sum(1 for r in deterministic if r.divergent)}/"
        f"{len(deterministic)} deterministic scenarios and "
        f"{len(organic_lossy)}/{len(organic_with_multi_flip)} organic seeds "
        "with a multi-event prompt -- inconsistent across scenarios rather "
        "than a clean universal gap.",
    )


@dataclass(frozen=True)
class AnnotationExportPairingReport:
    """Full fixture report for SLM-235."""

    schema: str = "AnnotationExportPairingReportV1"
    matrix_set: str = MATRIX_SET
    matrix_version: str = MATRIX_VERSION
    experiment_id: str = EXPERIMENT_ID
    run_id: str = "slm235-annotation-export-pairing-consistency"
    status: str = "fixture"
    claim_class: str = "wiring"
    hypothesis: str = _HYPOTHESIS
    falsifier: str = _FALSIFIER
    results: tuple[ScenarioResult, ...] = field(default_factory=tuple)
    static_shared_default_path_audit: dict[str, Any] = field(default_factory=dict)
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
            "results": [r.to_dict() for r in self.results],
            "static_shared_default_path_audit": dict(self.static_shared_default_path_audit),
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
    def from_dict(cls, data: dict[str, Any]) -> "AnnotationExportPairingReport":
        results = tuple(
            ScenarioResult(
                scenario=FlipScenario(
                    name=str(r["scenario"]["name"]),
                    description=str(r["scenario"]["description"]),
                    prompts=tuple(
                        (str(p), tuple(str(x) for x in ratings))
                        for p, ratings in r["scenario"]["prompts"]
                    ),
                    is_negative_control=bool(r["scenario"].get("is_negative_control", False)),
                    seed=r["scenario"].get("seed"),
                ),
                n_prompts=int(r["n_prompts"]),
                n_annotations=int(r["n_annotations"]),
                incremental_pairs_in_file=int(r["incremental_pairs_in_file"]),
                batch_pairs_written=int(r["batch_pairs_written"]),
                pairs_retained=int(r["pairs_retained"]),
                pairs_lost=int(r["pairs_lost"]),
                prompts_with_multi_events=int(r["prompts_with_multi_events"]),
                prompts_with_loss=int(r["prompts_with_loss"]),
                divergent=bool(r["divergent"]),
            )
            for r in data.get("results", ())
        )
        return cls(
            schema=str(data.get("schema", "AnnotationExportPairingReportV1")),
            matrix_set=str(data.get("matrix_set", MATRIX_SET)),
            matrix_version=str(data.get("matrix_version", MATRIX_VERSION)),
            experiment_id=str(data.get("experiment_id", EXPERIMENT_ID)),
            run_id=str(data.get("run_id", EXPERIMENT_ID)),
            status=str(data.get("status", "fixture")),
            claim_class=str(data.get("claim_class", "wiring")),
            hypothesis=str(data.get("hypothesis", _HYPOTHESIS)),
            falsifier=str(data.get("falsifier", _FALSIFIER)),
            results=results,
            static_shared_default_path_audit=dict(data.get("static_shared_default_path_audit", {})),
            gate_hash=str(data.get("gate_hash", "")),
            disposition=str(data.get("disposition", "inconclusive")),
            disposition_rationale=str(data.get("disposition_rationale", "")),
            honest_caveats=tuple(data.get("honest_caveats", _HONEST_CAVEATS)),
            version_stamp=dict(data.get("version_stamp", {})),
            timestamp=str(data.get("timestamp", _now())),
        )


def run_pairing_consistency_fixture(
    *,
    scenarios: list[FlipScenario] | None = None,
    run_id: str | None = None,
) -> AnnotationExportPairingReport:
    """Run every scenario through the real persist_annotation (live) then
    export_to_preference_pairs (batch) pipeline and compare pair sets."""
    scenarios = scenarios if scenarios is not None else build_default_scenarios()

    with tempfile.TemporaryDirectory(prefix="slm235-aep0-01-") as tmp:
        tmp_dir = Path(tmp)
        results = [_run_scenario(scenario, tmp_dir) for scenario in scenarios]

    static_audit = _static_shared_default_path_audit()
    disposition, rationale = _resolve_disposition(results, static_audit)

    payload = {
        "row_digests": sorted(_digest(r.to_dict()) for r in results),
        "static_audit": static_audit,
    }
    gate_hash = _sha256(_canonical_json(payload))

    return AnnotationExportPairingReport(
        run_id=run_id or f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        results=tuple(results),
        static_shared_default_path_audit=static_audit,
        gate_hash=gate_hash,
        disposition=disposition,
        disposition_rationale=rationale,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm235_annotation_export_pairing_consistency",
        ),
    )


def render_markdown(report: AnnotationExportPairingReport) -> str:
    lines = [
        f"# SLM-235 (AEP0-01): annotation export pairing-mechanism consistency probe ({report.run_id})",
        "",
        f"**Matrix set:** `{report.matrix_set}`",
        f"**Version:** `{report.matrix_version}`",
        f"**Status:** {report.status}",
        f"**Claim class:** {report.claim_class}",
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
        "## Static shared-default-path audit",
        "",
        "| field | value |",
        "| --- | --- |",
    ]
    for key, value in sorted(report.static_shared_default_path_audit.items()):
        lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Honest caveats",
        "",
        *(f"- {c}" for c in report.honest_caveats),
        "",
        "## Per-scenario results",
        "",
        "| scenario | prompts | annotations | incremental pairs | batch pairs | retained | lost | divergent | control |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        lines.append(
            f"| {r.scenario.name} | {r.n_prompts} | {r.n_annotations} | "
            f"{r.incremental_pairs_in_file} | {r.batch_pairs_written} | "
            f"{r.pairs_retained} | {r.pairs_lost} | {r.divergent} | "
            f"{r.scenario.is_negative_control} |"
        )
    lines += [
        "",
        "## Scenario descriptions",
        "",
    ]
    for r in report.results:
        lines.append(f"- **{r.scenario.name}**: {r.scenario.description}")
    lines += [
        "",
        "## No-go for promotion",
        "",
        "This report is wiring/fixture evidence only. It does not change "
        "`persist_annotation`, `maybe_append_preference_pair`, "
        "`export_to_preference_pairs`, `write_pairs`, or any CLI default, "
        "does not train a preference model, and makes no ship or gate "
        "claim. It documents a concrete mechanism inconsistency between the "
        "live and batch preference-pairing code paths in the annotation "
        "export pipeline, as a candidate for a future, separately reviewed "
        "hardening change (never implemented here).",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python -m scripts.run_slm235_annotation_export_pairing_consistency --mode plan-only",
        "python -m scripts.run_slm235_annotation_export_pairing_consistency --mode fixture",
        "```",
        "",
    ]
    return "\n".join(lines)
