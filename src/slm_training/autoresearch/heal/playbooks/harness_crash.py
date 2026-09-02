"""Code-class playbook: triage a crashed arm into a typed ``repair_harness``.

Until this playbook existed the ``code`` class had no executor at all — every
``harness_failure:<arm>:experiment_failed`` blocker escalated as
``no_matching_playbook`` while the driver's residual check read the same
crash as a soft wall timeout and dropped the blocker (cycles c536..c543).

This playbook never fixes code. It performs the bounded, deterministic part
of a crash triage so the owner skill starts from evidence, not a bare
``experiment_failed``:

1. capture the failing arm's exit code, stderr tail and traceback from the
   campaign's ``sdlc_delivery.json``, ``artifacts/outcomes/*.json``, per-arm
   run logs and the loop's driver/supervisor logs;
2. name the failing module and map it to a ``harness_family`` from the action
   schema;
3. emit a typed ``AutotrainActionV1(kind="repair_harness")`` carrying the
   traceback (written next to the loop's heal ledger, never into a handoff);
4. run that module's test file once (``pytest -x -q``, under the run cap;
   skipped when no test file exists) and record the result;
5. write a receipt with outcome ``attempted`` — never ``healed`` — so the pass
   is honest and no driver rewrite can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from slm_training.autoresearch.heal.classify import (
    HARNESS_CRASH_REASON_RE,
    TIMEOUT_EXIT_CODE,
)
from slm_training.autoresearch.heal.escalation import blocker_fingerprint
from slm_training.autoresearch.heal.schemas import (
    HealAttemptReceiptV1,
    HealPlanV1,
    HealStepResultV1,
    HealStepV1,
    HealVerifyV1,
)
from slm_training.autoresearch.schemas import AutotrainActionV1, utc_now
from slm_training.harness_core.bounded_process import run_bounded_process
from slm_training.levers import KILL_GRACE_SECONDS, MAX_RUN_SECONDS

__all__ = [
    "PLAYBOOK",
    "PLAYBOOK_ID",
    "CrashEvidence",
    "build_repair_action",
    "capture_crash_evidence",
    "execute",
    "extract_traceback",
    "harness_family_for_module",
    "mirrored_test_file",
]

PLAYBOOK_ID = "harness_crash/v1"

EVIDENCE_DIRNAME = "heal_harness_crash"

_STDERR_TAIL_CHARS = 4000
_TRACEBACK_CHARS = 3000
_REASON_TRACEBACK_CHARS = 1200

_TRACEBACK_HEAD = "Traceback (most recent call last):"

# Frames inside the interpreter / site-packages never name the failing
# repo module; the last repo frame does.
_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')
_MODULE_ERROR_RE = re.compile(
    r"No module named ['\"]?([A-Za-z0-9_.]+)['\"]?", re.IGNORECASE
)

#: ``src/slm_training/harnesses/<dir>`` → ``HarnessFamily``. Directories
#: without a family of their own fold into the family that owns their
#: ship-gate (eval/quantization → model_build; retrieval/reasoning →
#: quality), matching the driver's ``model_build`` default.
_HARNESS_DIR_FAMILY: dict[str, str] = {
    "annotations": "annotations",
    "distill": "distill",
    "experiments": "experiments",
    "model_build": "model_build",
    "preference": "preference",
    "quality": "quality",
    "rl": "rl",
    "test_data": "test_data",
    "train_data": "train_data",
    "eval": "model_build",
    "quantization": "model_build",
    "representations": "model_build",
    "retrieval": "quality",
    "reasoning": "quality",
}

_SCRIPT_FAMILY: tuple[tuple[str, str], ...] = (
    ("build_train_data", "train_data"),
    ("build_test_data", "test_data"),
    ("run_autotrain", "autoresearch"),
    ("autoresearch", "autoresearch"),
    ("train_model", "model_build"),
    ("evaluate_model", "model_build"),
    ("promote", "model_build"),
    ("distill", "distill"),
    ("preference", "preference"),
    ("rl_", "rl"),
    ("annotat", "annotations"),
    ("quality", "quality"),
)

_DEFAULT_FAMILY = "model_build"


@dataclass
class CrashEvidence:
    """What the crash left behind, gathered from every place it could land."""

    campaign_id: str = ""
    arm_id: str = ""
    exit_code: int | None = None
    reasons: tuple[str, ...] = ()
    stderr_tail: str = ""
    traceback: str = ""
    module: str = ""
    harness_family: str = _DEFAULT_FAMILY
    sources: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def is_timeout(self) -> bool:
        return self.exit_code == TIMEOUT_EXIT_CODE

    def to_dict(self) -> dict:
        return {
            "campaign_id": self.campaign_id,
            "arm_id": self.arm_id,
            "exit_code": self.exit_code,
            "reasons": list(self.reasons),
            "stderr_tail": self.stderr_tail,
            "traceback": self.traceback,
            "module": self.module,
            "harness_family": self.harness_family,
            "sources": list(self.sources),
            "notes": list(self.notes),
        }


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_tail(path: Path, chars: int = _STDERR_TAIL_CHARS) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-chars:]


def extract_traceback(text: str) -> str:
    """The last Python traceback block in ``text`` (empty when none)."""
    if not text:
        return ""
    start = text.rfind(_TRACEBACK_HEAD)
    if start < 0:
        return ""
    block = text[start:]
    lines = block.splitlines()
    out: list[str] = []
    for line in lines:
        out.append(line)
        stripped = line.strip()
        # The exception line ends the block: not indented, not a frame /
        # source line, and past the header.
        if (
            len(out) > 1
            and stripped
            and not line.startswith((" ", "\t"))
            and not stripped.startswith(_TRACEBACK_HEAD)
        ):
            break
    return "\n".join(out)[-_TRACEBACK_CHARS:]


def _repo_relative(path_text: str) -> str:
    text = path_text.replace("\\", "/")
    for anchor in ("/src/slm_training/", "/scripts/", "/slm_training/"):
        idx = text.rfind(anchor)
        if idx >= 0:
            rel = text[idx + 1 :]
            return rel[len("src/") :] if rel.startswith("src/") else rel
    return ""


def failing_module_from_traceback(traceback: str) -> str:
    """Repo-relative path of the last repo frame, else the missing module."""
    frames = [
        _repo_relative(m.group(1)) for m in _FRAME_RE.finditer(traceback)
    ]
    repo_frames = [f for f in frames if f]
    if repo_frames:
        return repo_frames[-1]
    missing = _MODULE_ERROR_RE.search(traceback)
    if missing:
        return missing.group(1)
    return ""


def harness_family_for_module(module: str) -> str:
    """Map a repo module path (or dotted name) to an action ``harness_family``."""
    text = str(module).replace("\\", "/").replace(".", "/").lower()
    if not text:
        return _DEFAULT_FAMILY
    parts = [p for p in text.split("/") if p]
    if "harnesses" in parts:
        idx = parts.index("harnesses")
        if idx + 1 < len(parts):
            return _HARNESS_DIR_FAMILY.get(parts[idx + 1], _DEFAULT_FAMILY)
    if "autoresearch" in parts:
        return "autoresearch"
    if parts[0] == "scripts" and len(parts) > 1:
        name = "/".join(parts[1:])
        for needle, family in _SCRIPT_FAMILY:
            if needle in name:
                return family
    for needle, family in _SCRIPT_FAMILY:
        if needle in text:
            return family
    return _DEFAULT_FAMILY


def mirrored_test_file(module: str, cwd: Path) -> Path | None:
    """The mirrored ``tests/`` file for a repo module, when one exists.

    Mirrors the repo convention: ``slm_training/a/b/c.py`` →
    ``tests/test_a/b/test_c.py`` or ``tests/test_a/test_b_c.py``;
    ``scripts/x.py`` → ``tests/test_scripts/test_x.py``.
    """
    text = str(module).replace("\\", "/")
    if text.startswith("src/"):
        text = text[len("src/") :]
    if text.endswith(".py"):
        text = text[: -len(".py")]
    elif "/" not in text and "." in text:
        text = text.replace(".", "/")
    parts = [p for p in text.split("/") if p and p != "__init__"]
    if not parts:
        return None
    candidates: list[Path] = []
    if parts[0] == "scripts" and len(parts) >= 2:
        candidates.append(cwd / "tests" / "test_scripts" / f"test_{parts[-1]}.py")
    elif parts[0] == "slm_training" and len(parts) >= 3:
        pkg, *middle, mod = parts[1:]
        base = cwd / "tests" / f"test_{pkg}"
        if middle:
            candidates.append(base.joinpath(*middle) / f"test_{mod}.py")
            candidates.append(base / f"test_{'_'.join(middle)}_{mod}.py")
            candidates.append(base / f"test_{middle[-1]}_{mod}.py")
        candidates.append(base / f"test_{mod}.py")
    elif parts[0] == "slm_training" and len(parts) == 2:
        candidates.append(cwd / "tests" / f"test_{parts[1]}.py")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _arm_from_reason(reason: str) -> str:
    match = re.search(r"harness_failure:([^:\s]+):experiment_failed", reason)
    return match.group(1) if match else ""


def _iter_outcome_files(campaign_dir: Path) -> Iterable[Path]:
    outcomes = campaign_dir / "artifacts" / "outcomes"
    if outcomes.is_dir():
        yield from sorted(outcomes.glob("*.json"))


def _arm_log_files(campaign_dir: Path, arm_id: str) -> list[Path]:
    found: list[Path] = []
    for base in (campaign_dir / "runs", campaign_dir / "artifacts" / "runs"):
        if not base.is_dir():
            continue
        for pattern in ("*stderr*", "*.err", "*.log"):
            for path in sorted(base.rglob(pattern)):
                if not path.is_file():
                    continue
                if arm_id and arm_id not in str(path):
                    continue
                found.append(path)
    return found


def capture_crash_evidence(
    campaign_dir: Path,
    *,
    arm_id: str | None = None,
    extra_logs: Sequence[Path] = (),
) -> CrashEvidence:
    """Gather exit code, stderr tail and traceback for the failing arm.

    Sources, in evidence order: ``sdlc_delivery.json`` (arm exits + reasons),
    ``artifacts/outcomes/<arm>.json`` (exit code, error, stage stderr),
    per-arm run logs under ``runs/``, then any ``extra_logs`` (driver /
    supervisor logs) as a last resort.
    """
    campaign_dir = Path(campaign_dir)
    evidence = CrashEvidence(campaign_id=campaign_dir.name)
    sources: list[str] = []
    delivery = _read_json(campaign_dir / "sdlc_delivery.json")
    reasons = tuple(str(r) for r in (delivery.get("reasons") or []))
    evidence.reasons = reasons
    exits = delivery.get("arm_exits") or {}
    if delivery:
        sources.append("sdlc_delivery.json")
    if arm_id is None or not arm_id:
        for reason in reasons:
            arm_id = _arm_from_reason(reason)
            if arm_id:
                break
    if not arm_id and isinstance(exits, dict):
        crashed = [
            k
            for k, v in exits.items()
            if _int_or_none(v) not in (None, 0, TIMEOUT_EXIT_CODE)
        ]
        arm_id = crashed[0] if crashed else ""
    evidence.arm_id = str(arm_id or "")
    if isinstance(exits, dict) and evidence.arm_id in exits:
        evidence.exit_code = _int_or_none(exits[evidence.arm_id])

    stderr_chunks: list[str] = []
    for path in _iter_outcome_files(campaign_dir):
        payload = _read_json(path)
        eid = str(payload.get("experiment_id") or path.stem)
        if evidence.arm_id and eid != evidence.arm_id and path.stem != evidence.arm_id:
            continue
        if not evidence.arm_id:
            evidence.arm_id = eid
        if evidence.exit_code is None and payload.get("exit_code") is not None:
            evidence.exit_code = _int_or_none(payload.get("exit_code"))
        error = str(payload.get("error") or "")
        if error:
            stderr_chunks.append(error)
        for stage in payload.get("stage_telemetry") or []:
            if not isinstance(stage, dict):
                continue
            text = str(stage.get("stderr") or "")
            if text:
                stderr_chunks.append(text)
            if evidence.exit_code is None and stage.get("exit_code") is not None:
                evidence.exit_code = _int_or_none(stage.get("exit_code"))
        sources.append(str(path.relative_to(campaign_dir)))
    for path in _arm_log_files(campaign_dir, evidence.arm_id):
        text = _read_tail(path)
        if text:
            stderr_chunks.append(text)
            sources.append(str(path.relative_to(campaign_dir)))
    for path in extra_logs:
        path = Path(path)
        if not path.is_file():
            continue
        text = _read_tail(path)
        if text and (not evidence.arm_id or evidence.arm_id in text or _TRACEBACK_HEAD in text):
            stderr_chunks.append(text)
            sources.append(str(path))

    combined = "\n".join(stderr_chunks)
    evidence.stderr_tail = combined[-_STDERR_TAIL_CHARS:]
    evidence.traceback = extract_traceback(combined)
    evidence.module = failing_module_from_traceback(evidence.traceback)
    if not evidence.module:
        evidence.module = failing_module_from_traceback(evidence.stderr_tail)
    evidence.harness_family = harness_family_for_module(evidence.module)
    evidence.sources = tuple(sources)
    if not evidence.traceback:
        evidence.notes.append("no_traceback_captured")
    if not evidence.module:
        evidence.notes.append("failing_module_unresolved")
    return evidence


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def build_repair_action(
    evidence: CrashEvidence, *, campaign_id: str
) -> AutotrainActionV1:
    """A typed ``repair_harness`` action carrying the captured traceback."""
    arm = evidence.arm_id or "unknown-arm"
    exit_text = "unknown" if evidence.exit_code is None else str(evidence.exit_code)
    traceback = evidence.traceback or evidence.stderr_tail
    traceback = traceback[-_REASON_TRACEBACK_CHARS:].strip()
    reason = (
        f"harness_failure:{arm}:experiment_failed exit={exit_text} "
        f"module={evidence.module or 'unresolved'} "
        f"family={evidence.harness_family}; owner-skill fix required "
        "(harness_crash playbook diagnosed, no auto-fix)"
    )
    if traceback:
        reason += "\ntraceback:\n" + traceback
    evidence_ids = [f"campaign:{campaign_id}", f"arm:{arm}"]
    evidence_ids.extend(f"source:{s}" for s in evidence.sources[:6])
    return AutotrainActionV1(
        kind="repair_harness",
        owner="improve-openui-harnesses",
        reason=reason,
        evidence_ids=tuple(evidence_ids),
        harness_family=evidence.harness_family,  # type: ignore[arg-type]
    )


def evidence_path(root: Path, loop_id: str, fingerprint: str) -> Path:
    return Path(root) / "loops" / loop_id / EVIDENCE_DIRNAME / f"{fingerprint[:16]}.json"


def _writes_allowed_for(
    target: Path, *, cwd: Path, extra: Sequence[str] = ()
) -> tuple[str, ...]:
    """Repo-relative allowlist for a step that writes under ``target``.

    ``outputs/`` is always included (loop roots live there by default); a
    target outside the worktree contributes nothing (git never sees it).
    """
    try:
        rel = Path(target).resolve().relative_to(Path(cwd).resolve()).as_posix()
    except ValueError:
        rel = ""
    allowed: list[str] = []
    if rel and rel != ".":
        allowed.append(f"{rel.rstrip('/')}/")
    allowed.append("outputs/")
    allowed.append(".pytest_cache/")
    allowed.extend(extra)
    return tuple(dict.fromkeys(allowed))


def _run_module_tests(test_file: Path | None, *, cwd: Path) -> HealStepResultV1:
    if test_file is None:
        empty = hashlib.sha256(b"").hexdigest()
        return HealStepResultV1(
            step_id="module_tests",
            returncode=0,
            outcome="skipped",
            stdout_sha256=empty,
            stderr_sha256=empty,
            tail="skipped: no mirrored test file for the failing module",
        )
    result = run_bounded_process(
        [sys.executable, "-m", "pytest", "-x", "-q", str(test_file)],
        interrupt_after_seconds=float(MAX_RUN_SECONDS - KILL_GRACE_SECONDS),
        kill_grace_seconds=float(KILL_GRACE_SECONDS),
        cwd=str(cwd),
    )
    tail = (result.stdout[-1000:] + "\n" + result.stderr[-1000:]).strip()[:2000]
    return HealStepResultV1(
        step_id="module_tests",
        returncode=-1 if result.returncode is None else int(result.returncode),
        outcome=result.outcome.value,
        duration_seconds=float(result.duration_seconds),
        stdout_sha256=hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        stderr_sha256=hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
        tail=tail,
    )


def execute(
    blocker: dict,
    *,
    cwd: Path,
    root: Path,
    loop_id: str,
    campaign_id: str,
    run_tests: bool = True,
    write_receipt: bool = True,
) -> HealAttemptReceiptV1:
    """Triage one crash blocker; returns (and by default persists) the receipt.

    The receipt outcome is always ``attempted``: the playbook diagnoses and
    emits the typed action but repairs nothing, so nothing is verified.
    """
    from slm_training.autoresearch.heal import write_heal_receipt

    cwd = Path(cwd)
    root = Path(root)
    kind = str(blocker.get("kind") or "repair_harness")
    reason = str(blocker.get("reason") or "")
    fingerprint = blocker_fingerprint(kind, reason)
    blocker_campaign = str(blocker.get("campaign_id") or campaign_id)
    campaign_dir = root / blocker_campaign
    arm_id = str(blocker.get("arm_id") or "") or _arm_from_reason(reason) or None
    extra_logs = [root / "loops" / loop_id / "supervisor.jsonl"]
    extra_logs.extend(Path(p) for p in (blocker.get("_logs") or ()))
    evidence = capture_crash_evidence(
        campaign_dir, arm_id=arm_id, extra_logs=extra_logs
    )
    if not evidence.reasons and reason:
        evidence.reasons = (reason,)
    if evidence.is_timeout:
        evidence.notes.append("arm_exit_124_is_a_timeout_not_a_crash")
    action = build_repair_action(evidence, campaign_id=blocker_campaign)
    test_file = mirrored_test_file(evidence.module, cwd) if evidence.module else None
    steps: list[HealStepResultV1] = []
    if run_tests:
        steps.append(_run_module_tests(test_file, cwd=cwd))
    else:
        steps.append(_run_module_tests(None, cwd=cwd))
    payload = {
        "schema_version": "heal_harness_crash_evidence/v1",
        "loop_id": loop_id,
        "campaign_id": blocker_campaign,
        "blocker_fingerprint": fingerprint,
        "recorded_at": utc_now(),
        "action": action.model_dump(mode="json"),
        "evidence": evidence.to_dict(),
        "module_test_file": str(test_file) if test_file else None,
        "module_tests": steps[-1].model_dump(mode="json"),
    }
    out_path = evidence_path(root, loop_id, fingerprint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    test_summary = (
        "module_tests=skipped"
        if steps[-1].outcome == "skipped"
        else f"module_tests={steps[-1].outcome}:rc={steps[-1].returncode}"
    )
    note = (
        f"diagnosed arm={evidence.arm_id or 'unknown'} exit={evidence.exit_code} "
        f"module={evidence.module or 'unresolved'} family={evidence.harness_family} "
        f"{test_summary} action={out_path}"
    )
    if evidence.notes:
        note += " notes=" + ",".join(evidence.notes)
    receipt = HealAttemptReceiptV1(
        loop_id=loop_id,
        campaign_id=blocker_campaign,
        playbook_id=PLAYBOOK_ID,
        plan_sha256=hashlib.sha256(
            json.dumps(payload["action"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        blocker_fingerprint=fingerprint,
        attempts_prior=int(blocker.get("_attempts_prior") or 0),
        step_results=tuple(steps),
        verify_result=None,
        outcome="attempted",
        note=note[:2000],
        recorded_at=utc_now(),
    )
    if write_receipt:
        write_heal_receipt(root, receipt)
    return receipt


class _HarnessCrashPlaybook:
    playbook_id = PLAYBOOK_ID
    handles = frozenset({"code"})

    def matches(self, blocker: dict) -> bool:
        kind = str(blocker.get("kind") or "")
        reason = str(blocker.get("reason") or "")
        if kind != "repair_harness":
            return False
        if HARNESS_CRASH_REASON_RE.search(reason):
            return True
        # Any other code-class repair_harness (repo-internal import error…)
        # still deserves the same traceback capture before escalation.
        return bool(reason)

    def execute(
        self,
        blocker: dict,
        *,
        cwd: Path,
        root: Path,
        loop_id: str,
        campaign_id: str,
    ) -> HealAttemptReceiptV1:
        return execute(
            blocker, cwd=cwd, root=root, loop_id=loop_id, campaign_id=campaign_id
        )

    def plan(self, blocker: dict, *, cwd: Path) -> HealPlanV1 | None:
        """Runner-compatible plan: the diagnosis step plus an honest verify.

        The verify probe always exits non-zero: a code crash has no
        self-verifiable repair, so a runner that executes this plan records
        ``verify_failed`` (never ``healed``) alongside the diagnosis file the
        step wrote. Runners that support ``execute`` should prefer it and
        record ``attempted`` instead.
        """
        root = blocker.get("_root")
        loop_id = str(blocker.get("_loop_id") or "")
        if root is None or not loop_id:
            return None
        root = Path(root)
        campaign_id = str(blocker.get("campaign_id") or "")
        if not campaign_id:
            return None
        reason = str(blocker.get("reason") or "")
        writes_allowed = _writes_allowed_for(
            evidence_path(root, loop_id, "0" * 16).parent, cwd=Path(cwd)
        )
        argv = (
            sys.executable,
            "-m",
            "slm_training.autoresearch.heal.playbooks.harness_crash",
            "--root",
            str(root),
            "--loop-id",
            loop_id,
            "--campaign-id",
            campaign_id,
            "--reason",
            reason,
            "--no-receipt",
        )
        return HealPlanV1(
            playbook_id=self.playbook_id,
            blocker_fingerprint=blocker_fingerprint(
                str(blocker.get("kind") or ""), reason
            ),
            blocker_class="code",
            steps=(
                HealStepV1(
                    step_id="diagnose_crash",
                    argv=argv,
                    cwd="",
                    timeout_seconds=int(MAX_RUN_SECONDS),
                    writes_allowed=writes_allowed,
                ),
            ),
            verify=HealVerifyV1(
                argv=(
                    sys.executable,
                    "-c",
                    (
                        "import sys; print('harness_crash: code-class crash "
                        "triaged; owner-skill fix required, nothing verified'); "
                        "sys.exit(1)"
                    ),
                ),
                cwd="",
                timeout_seconds=30,
            ),
        )


PLAYBOOK = _HarnessCrashPlaybook()


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--loop-id", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--reason", default="")
    parser.add_argument("--arm", default="")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--no-tests", action="store_true")
    parser.add_argument("--no-receipt", action="store_true")
    args = parser.parse_args(argv)
    blocker = {
        "kind": "repair_harness",
        "reason": args.reason,
        "campaign_id": args.campaign_id,
        "arm_id": args.arm,
    }
    receipt = execute(
        blocker,
        cwd=args.cwd,
        root=args.root,
        loop_id=args.loop_id,
        campaign_id=args.campaign_id,
        run_tests=not args.no_tests,
        write_receipt=not args.no_receipt,
    )
    print(receipt.model_dump_json())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI shim
    sys.exit(_main())
