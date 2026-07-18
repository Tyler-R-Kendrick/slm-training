"""Allowlisted background-job runner for the local control plane.

Security boundary: the web app must only ever launch a **fixed set of known
scripts** with **validated arguments**. Arbitrary command execution is never
possible — there is no ``shell=True``, no string interpolation, and every
parameter is checked against a typed rule (:class:`Choice` / :class:`Slug` /
:class:`IntRange` / :class:`Flag` / :class:`PathTemplate`) before it becomes a
list-form ``argv`` element.

Execution model: a single serial ``asyncio`` worker drains a queue and runs one
job at a time via ``subprocess.Popen(start_new_session=True)`` with stdout/stderr
redirected straight to an append-only log file (no pump thread). Job history is
mirrored to ``outputs/jobs/<id>/meta.json`` so a restarted server can re-list it.
Live logs stream to the browser over SSE. Heavy GPU trains are intentionally
absent from the allowlist — they are dispatched to HF Jobs / pods by the
``dispatch``-kind wrappers, never run in-process.

This module stays import-light (no torch) so it never blocks the Vercel entry.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

TERMINAL = {"succeeded", "failed", "cancelled"}


# --------------------------------------------------------------------------- #
# Parameter rules — the validation layer that makes exec safe.
# --------------------------------------------------------------------------- #
class ParamRule:
    def coerce(self, value: Any) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def render(self, name: str, value: Any) -> list[str]:
        return [f"--{name.replace('_', '-')}", self.coerce(value)]


class Choice(ParamRule):
    def __init__(self, *allowed: str) -> None:
        self.allowed = {str(a) for a in allowed}

    def coerce(self, value: Any) -> str:
        s = str(value)
        if s not in self.allowed:
            raise ValueError(f"value {s!r} not in allowed {sorted(self.allowed)}")
        return s


class Slug(ParamRule):
    def __init__(self, pattern: str = r"^[A-Za-z0-9._,-]{1,64}$") -> None:
        self.re = re.compile(pattern)

    def coerce(self, value: Any) -> str:
        s = str(value)
        if not self.re.match(s):
            raise ValueError(f"value {s!r} fails pattern {self.re.pattern}")
        return s


class IntRange(ParamRule):
    def __init__(self, lo: int, hi: int) -> None:
        self.lo, self.hi = lo, hi

    def coerce(self, value: Any) -> str:
        n = int(value)
        if not (self.lo <= n <= self.hi):
            raise ValueError(f"{n} outside [{self.lo}, {self.hi}]")
        return str(n)


class Flag(ParamRule):
    def coerce(self, value: Any) -> str:
        return "true" if value else "false"

    def render(self, name: str, value: Any) -> list[str]:
        return [f"--{name.replace('_', '-')}"] if value else []


class BooleanOptionalFlag(ParamRule):
    """Render argparse.BooleanOptionalAction values in either direction."""

    def coerce(self, value: Any) -> str:
        return "true" if value else "false"

    def render(self, name: str, value: Any) -> list[str]:
        prefix = "" if value else "no-"
        return [f"--{prefix}{name.replace('_', '-')}"]


class PathTemplate(ParamRule):
    """Render a filesystem flag from a slug-validated token — no path escape."""

    def __init__(self, flag: str, template: str, inner: ParamRule | None = None) -> None:
        self.flag = flag
        self.template = template
        self.inner = inner or Slug()

    def coerce(self, value: Any) -> str:
        return self.template.format(value=self.inner.coerce(value))

    def render(self, name: str, value: Any) -> list[str]:
        return [self.flag, self.coerce(value)]


@dataclass(frozen=True)
class JobSpec:
    module: str
    kind: str = "local"  # "local" | "dispatch"
    positional: tuple[str, ...] = ()
    params: dict[str, ParamRule] = field(default_factory=dict)
    summary: str = ""

    def render_argv(self, values: dict[str, Any]) -> list[str]:
        argv = [sys.executable, "-u", "-m", self.module]
        for name in self.positional:
            if name not in values:
                raise ValueError(f"missing required parameter: {name}")
            argv.append(self.params[name].coerce(values[name]))
        for name, rule in self.params.items():
            if name in self.positional:
                continue
            if name not in values or values[name] in (None, ""):
                continue
            if values[name] is False and not isinstance(rule, BooleanOptionalFlag):
                continue
            argv.extend(rule.render(name, values[name]))
        return argv


# --------------------------------------------------------------------------- #
# The allowlist. Only these scripts can run, only with these params.
# --------------------------------------------------------------------------- #
JOB_SPECS: dict[str, JobSpec] = {
    "build_train_data": JobSpec(
        "scripts.build_train_data",
        summary="Build a versioned training corpus",
        params={
            "source": Choice(
                "all",
                "rico",
                "fixture",
                "both",
                "awwwards",
                "rico+awwwards",
                "existing",
                "programspec",
                "language_contract",
                "deconstruct",
                "render",
                "integrated",
            ),
            "version": Slug(),
            "base_version": PathTemplate(
                "--derive-from", "outputs/data/train/{value}/records.jsonl"
            ),
            "synthesizer": Choice("quality", "template", "layout", "frontier", "none"),
            "profile": Choice("strict", "permissive"),
            "namespace_augment": Flag(),
            "edit_derivatives": BooleanOptionalFlag(),
            "repairs_per_program": IntRange(0, 8),
            "fuzzy_dedup": Flag(),
            "curriculum": Flag(),
        },
    ),
    "mine_rejected_preferences": JobSpec(
        "scripts.mine_rejected_preferences",
        summary="Mine preference pairs from a dataset's rejected-record ledger",
        params={
            "dataset": Slug(),
            "version": Slug(),
        },
    ),
    "build_test_data": JobSpec(
        "scripts.build_test_data",
        summary="Build disjoint eval suites (with leakage checks)",
        params={
            "source": Choice("both", "rico", "fixture", "awwwards"),
            "version": Slug(),
            "train_version": PathTemplate(
                "--train-manifest", "outputs/data/train/{value}/manifest.json"
            ),
        },
    ),
    "evaluate_model": JobSpec(
        "scripts.evaluate_model",
        summary="Evaluate a checkpoint (optionally with ship gates)",
        params={
            "test_version": PathTemplate("--test-dir", "outputs/data/eval/{value}"),
            "run_id": Slug(),
            "suite": Choice("smoke", "held_out", "adversarial", "ood", "rico_held"),
            "ship_gates": Flag(),
        },
    ),
    "run_quality_matrix": JobSpec(
        "scripts.run_quality_matrix",
        summary="Run the quality experiment matrix",
        params={
            "matrix": Choice("v2", "v3", "v4", "v5", "v6", "v7"),
            "only": Slug(),
            "steps": IntRange(1, 5000),
            "device": Choice("cpu", "cuda", "mps"),
            "context_backend": Choice("scratch", "hf"),
            "scratch_control": Flag(),
        },
    ),
    "run_grammar_matrix": JobSpec(
        "scripts.run_grammar_matrix",
        summary="Run the grammar (X-series) matrix",
        params={
            "only": Slug(),
            "steps": IntRange(1, 5000),
            "device": Choice("cpu", "cuda", "mps"),
        },
    ),
    "run_perf_matrix": JobSpec(
        "scripts.run_perf_matrix",
        summary="Run the performance (P/Q/R) matrix",
        params={
            "only": Slug(),
            "device": Choice("cpu", "cuda", "mps"),
            "suite": Choice("smoke", "held_out"),
        },
    ),
    "run_phase_pipeline": JobSpec(
        "scripts.run_phase_pipeline",
        summary="Run the phase A/B/C pipeline",
        params={
            "steps": IntRange(1, 5000),
            "device": Choice("cpu", "cuda", "mps"),
        },
    ),
    "model_cycle": JobSpec(
        "scripts.model_cycle",
        summary="Lineage lifecycle (promote / deploy / evaluate)",
        positional=("subcommand",),
        params={
            "subcommand": Choice("promote", "deploy", "evaluate"),
            "run_id": Slug(),
            "track": Choice("twotower", "causal_lm"),
        },
    ),
    "sync_checkpoints": JobSpec(
        "scripts.sync_checkpoints",
        summary="Sync a run's checkpoints to the HF bucket",
        params={
            "run_dir": PathTemplate("--run-dir", "outputs/runs/{value}"),
            "ensure_bucket": Flag(),
            "dry_run": Flag(),
        },
    ),
    # Dispatch-kind: kicks off a remote GPU job; the local process is only the
    # dispatcher/monitor and returns quickly. Heavy training never runs here.
    "hf_jobs_train": JobSpec(
        "scripts.hf_jobs_train",
        kind="dispatch",
        summary="Dispatch a full train to HF managed Jobs",
        params={"run_id": Slug(), "steps": IntRange(1, 100000), "dry_run": Flag()},
    ),
    "remote_train": JobSpec(
        "scripts.remote_train",
        kind="dispatch",
        summary="Dispatch a full train to a remote GPU pod over SSH",
        params={
            "host": Slug(r"^[A-Za-z0-9._-]{1,255}$"),
            "run_id": Slug(),
            "steps": IntRange(1, 100000),
            "dry_run": Flag(),
        },
    ),
}


def catalog() -> list[dict[str, Any]]:
    """Describe the allowlist for the UI (params + kinds)."""
    out: list[dict[str, Any]] = []
    for key, spec in JOB_SPECS.items():
        params = {}
        for name, rule in spec.params.items():
            info: dict[str, Any] = {"type": type(rule).__name__}
            if isinstance(rule, Choice):
                info["choices"] = sorted(rule.allowed)
            elif isinstance(rule, IntRange):
                info["min"], info["max"] = rule.lo, rule.hi
            params[name] = info
        out.append(
            {
                "job": key,
                "kind": spec.kind,
                "summary": spec.summary,
                "positional": list(spec.positional),
                "params": params,
            }
        )
    return out


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Job:
    id: str
    job_key: str
    argv: list[str]
    status: str = "queued"
    kind: str = "local"
    pid: int | None = None
    returncode: int | None = None
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    ended_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JobRegistry:
    """Serial background-job registry. Instantiated only when execution is on."""

    def __init__(self, root: Path | str = Path("."), *, concurrency: int = 1) -> None:
        self.repo_root = Path(root)
        self.jobs_dir = self.repo_root / "outputs" / "jobs"
        self.concurrency = max(1, int(concurrency))
        self.jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen[bytes]] = {}
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        self._load_history()
        for _ in range(self.concurrency):
            self._workers.append(asyncio.create_task(self._worker()))

    async def shutdown(self) -> None:
        for job_id, proc in list(self._procs.items()):
            if proc.poll() is None:
                self._terminate(proc)
        for task in self._workers:
            task.cancel()

    def _load_history(self) -> None:
        if not self.jobs_dir.exists():
            return
        for meta in sorted(self.jobs_dir.glob("*/meta.json")):
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                job = Job(**data)
                # A job left "running" by a killed server is no longer alive.
                if job.status in {"running", "queued"}:
                    job.status = "failed"
                self.jobs[job.id] = job
            except (OSError, ValueError, TypeError):
                continue

    # -- submission ---------------------------------------------------------
    def submit(self, job_key: str, params: dict[str, Any]) -> Job:
        spec = JOB_SPECS.get(job_key)
        if spec is None:
            raise KeyError(job_key)
        argv = spec.render_argv(params)  # raises ValueError on bad params
        job = Job(id=uuid.uuid4().hex[:12], job_key=job_key, argv=argv, kind=spec.kind)
        self.jobs[job.id] = job
        self._write_meta(job)
        self._queue.put_nowait(job.id)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            job.status = "cancelling"
            self._terminate(proc)
        elif job.status not in TERMINAL:
            job.status = "cancelled"
        self._write_meta(job)
        return job

    def list(self) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)

    # -- worker -------------------------------------------------------------
    async def _worker(self) -> None:
        while True:
            job_id = await self._queue.get()
            job = self.jobs.get(job_id)
            if job is None or job.status in TERMINAL or job.status == "cancelling":
                if job and job.status == "cancelling":
                    job.status = "cancelled"
                    self._write_meta(job)
                continue
            await self._run(job)

    async def _run(self, job: Job) -> None:
        job.status = "running"
        job.started_at = _now()
        log_path = self.jobs_dir / job.id / "log.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_meta(job)
        try:
            with open(log_path, "ab", buffering=0) as log:
                proc = subprocess.Popen(  # noqa: S603 - argv is allowlist-rendered
                    job.argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.repo_root),
                    start_new_session=True,
                )
                self._procs[job.id] = proc
                job.pid = proc.pid
                self._write_meta(job)
                rc = await asyncio.get_running_loop().run_in_executor(None, proc.wait)
        except OSError as exc:
            self._append_log(log_path, f"[job runner error] {exc}\n")
            job.status = "failed"
            job.ended_at = _now()
            self._write_meta(job)
            self._procs.pop(job.id, None)
            return
        job.returncode = rc
        if job.status == "cancelling":
            job.status = "cancelled"
        else:
            job.status = "succeeded" if rc == 0 else "failed"
        job.ended_at = _now()
        self._procs.pop(job.id, None)
        self._write_meta(job)

    # -- streaming ----------------------------------------------------------
    async def stream(self, job_id: str, *, poll: float = 0.25) -> AsyncIterator[str]:
        job = self.jobs.get(job_id)
        if job is None:
            yield _sse("error", {"error": "unknown job"})
            return
        log_path = self.jobs_dir / job_id / "log.txt"
        yield _sse("status", job.to_dict())
        pos = 0
        while True:
            chunk, pos = _read_from(log_path, pos)
            if chunk:
                for line in chunk.splitlines():
                    yield _sse("log", {"line": line})
            job = self.jobs.get(job_id)
            if job is None or job.status in TERMINAL:
                # Final drain, then terminal status.
                chunk, pos = _read_from(log_path, pos)
                for line in chunk.splitlines():
                    yield _sse("log", {"line": line})
                yield _sse("status", job.to_dict() if job else {"status": "unknown"})
                return
            await asyncio.sleep(poll)

    def tail(self, job_id: str, *, lines: int = 200) -> list[str]:
        log_path = self.jobs_dir / job_id / "log.txt"
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        return text.splitlines()[-lines:]

    # -- helpers ------------------------------------------------------------
    def _write_meta(self, job: Job) -> None:
        meta = self.jobs_dir / job.id / "meta.json"
        meta.parent.mkdir(parents=True, exist_ok=True)
        try:
            meta.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")
        except OSError:
            pass

    @staticmethod
    def _append_log(path: Path, text: str) -> None:
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(text)
        except OSError:
            pass

    @staticmethod
    def _terminate(proc: subprocess.Popen[bytes]) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass


def _read_from(path: Path, pos: int) -> tuple[str, int]:
    try:
        with open(path, "rb") as handle:
            handle.seek(pos)
            data = handle.read()
            return data.decode("utf-8", "replace"), handle.tell()
    except OSError:
        return "", pos


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
