"""Pinned, isolated deep-research implementations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from slm_training.autoresearch.schemas import (
    CampaignSpec,
    EvidenceSnapshot,
    OpenDeepResearchConfig,
    OpenResearcherConfig,
    ResearchRequest,
    ResearcherRun,
    ResearchSource,
    utc_now,
)
from slm_training.lineage.records import canonical_json

OPEN_DEEP_RESEARCH_REVISION = "b764481fca7f0dbf00b2c70239bd97cea59d1059"
OPEN_RESEARCHER_REVISION = "785fd6ba5fcbc068daa4a2f07bbe0964f2983c86"
MAX_RESULT_BYTES = 2_000_000


@dataclass(frozen=True)
class ResearcherSpec:
    researcher_id: str
    upstream_repo: str
    upstream_revision: str
    config_model: type[OpenDeepResearchConfig] | type[OpenResearcherConfig]


RESEARCHERS = {
    "open-deep-research": ResearcherSpec(
        "open-deep-research",
        "https://github.com/langchain-ai/open_deep_research",
        OPEN_DEEP_RESEARCH_REVISION,
        OpenDeepResearchConfig,
    ),
    "open-researcher": ResearcherSpec(
        "open-researcher",
        "https://github.com/TIGER-AI-Lab/OpenResearcher",
        OPEN_RESEARCHER_REVISION,
        OpenResearcherConfig,
    ),
}


class Researcher(Protocol):
    def run(
        self,
        campaign: CampaignSpec,
        evidence: EvidenceSnapshot,
        sources: list[ResearchSource],
    ) -> ResearcherRun: ...


class IsolatedResearcher:
    """Invoke one reviewed upstream checkout without importing it in this process."""

    def __init__(
        self,
        spec: ResearcherSpec,
        *,
        checkout: Path | str,
        python: Path | str,
        worker: Path | str,
        config: dict[str, Any] | None = None,
        timeout_seconds: float = 180,
    ) -> None:
        self.spec = spec
        self.checkout = Path(checkout).resolve()
        self.python = Path(python).resolve()
        self.worker = Path(worker).resolve()
        self.config = spec.config_model.model_validate(config or {}).model_dump(
            mode="json"
        )
        if not 0 < timeout_seconds <= 180:
            raise ValueError("researcher timeout must be in (0, 180]")
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        campaign: CampaignSpec,
        evidence: EvidenceSnapshot,
        sources: list[ResearchSource],
    ) -> ResearcherRun:
        started_at = utc_now()
        started = time.monotonic()
        request = ResearchRequest(
            researcher_id=self.spec.researcher_id,
            upstream_repo=self.spec.upstream_repo,
            upstream_revision=self.spec.upstream_revision,
            campaign_id=campaign.campaign_id,
            evidence_snapshot_id=evidence.snapshot_id,
            prompt=_research_prompt(campaign, evidence, sources),
            config=self.config,
        )
        request_sha = hashlib.sha256(
            canonical_json(request.model_dump(mode="json")).encode("utf-8")
        ).hexdigest()
        error = self._preflight()
        if error:
            return self._result(
                request_sha,
                started_at,
                started,
                status="failed",
                error=error,
            )

        with tempfile.TemporaryDirectory(prefix="openui-researcher-") as raw_tmp:
            tmp = Path(raw_tmp)
            request_path = tmp / "request.json"
            result_path = tmp / "result.json"
            stdout_path = tmp / "stdout.log"
            stderr_path = tmp / "stderr.log"
            request_path.write_text(request.model_dump_json(indent=2), encoding="utf-8")
            command = [
                str(self.python),
                str(self.worker),
                "--backend",
                self.spec.researcher_id,
                "--checkout",
                str(self.checkout),
                "--request",
                str(request_path),
                "--output",
                str(result_path),
            ]
            try:
                with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                    completed = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        timeout=self.timeout_seconds,
                    )
            except subprocess.TimeoutExpired:
                return self._result(
                    request_sha,
                    started_at,
                    started,
                    status="failed",
                    error=f"researcher timed out after {self.timeout_seconds:g}s",
                )
            telemetry = {
                "command": command,
                "returncode": completed.returncode,
                **_file_identity("stdout", stdout_path),
                **_file_identity("stderr", stderr_path),
            }
            if completed.returncode != 0:
                return self._result(
                    request_sha,
                    started_at,
                    started,
                    status="failed",
                    error=f"researcher worker exited {completed.returncode}",
                    telemetry=telemetry,
                )
            if not result_path.is_file() or result_path.stat().st_size > MAX_RESULT_BYTES:
                return self._result(
                    request_sha,
                    started_at,
                    started,
                    status="failed",
                    error="researcher result is missing or exceeds the 2 MB limit",
                    telemetry=telemetry,
                )
            try:
                raw = json.loads(result_path.read_text(encoding="utf-8"))
                memo = str(raw["memo"])
                trace = raw.get("trace") or {}
                worker_telemetry = raw.get("telemetry") or {}
                if not isinstance(trace, dict) or not isinstance(worker_telemetry, dict):
                    raise TypeError("trace and telemetry must be JSON objects")
                if not memo.strip():
                    raise ValueError("researcher returned an empty memo")
                if len(memo) > 250_000:
                    raise ValueError("memo exceeds the 250,000 character limit")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return self._result(
                    request_sha,
                    started_at,
                    started,
                    status="failed",
                    error=f"invalid researcher result: {exc}",
                    telemetry=telemetry,
                )
            normalized_sources = _merge_sources(
                sources,
                _sources_from_urls(memo, self.spec.researcher_id),
                _sources_from_urls(canonical_json(trace), self.spec.researcher_id),
            )
            telemetry["worker"] = worker_telemetry
            return self._result(
                request_sha,
                started_at,
                started,
                status="completed",
                memo=memo,
                sources=normalized_sources,
                trace=trace,
                telemetry=telemetry,
            )

    def _preflight(self) -> str | None:
        for label, path in (
            ("checkout", self.checkout),
            ("python", self.python),
            ("worker", self.worker),
        ):
            if not path.exists():
                return f"researcher {label} does not exist: {path}"
        if not self.checkout.is_dir():
            return f"researcher checkout is not a directory: {self.checkout}"
        if not self.worker.is_file():
            return f"researcher worker is not a file: {self.worker}"
        if not self.python.is_file() or not os.access(self.python, os.X_OK):
            return f"researcher python is not executable: {self.python}"
        try:
            revision = subprocess.run(
                ["git", "-C", str(self.checkout), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"cannot inspect researcher checkout revision: {exc}"
        actual = revision.stdout.strip()
        if revision.returncode != 0:
            return f"researcher checkout is not a Git worktree: {self.checkout}"
        if actual != self.spec.upstream_revision:
            return (
                f"researcher revision mismatch: expected {self.spec.upstream_revision}, "
                f"found {actual}"
            )
        return None

    def _result(
        self,
        request_sha: str,
        started_at: str,
        started: float,
        *,
        status: Literal["completed", "failed"],
        memo: str = "",
        sources: tuple[ResearchSource, ...] = (),
        trace: dict[str, Any] | None = None,
        telemetry: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> ResearcherRun:
        details = dict(telemetry or {})
        details["duration_seconds"] = round(time.monotonic() - started, 6)
        return ResearcherRun(
            researcher_id=self.spec.researcher_id,
            upstream_repo=self.spec.upstream_repo,
            upstream_revision=self.spec.upstream_revision,
            request_sha256=request_sha,
            status=status,
            memo=memo,
            sources=sources,
            trace=trace or {},
            telemetry=details,
            error=error,
            started_at=started_at,
            finished_at=utc_now(),
        )


def get_researcher(
    researcher_id: str,
    *,
    checkout: Path | str,
    python: Path | str,
    worker: Path | str,
    config: dict[str, Any] | None = None,
    timeout_seconds: float = 180,
) -> IsolatedResearcher:
    try:
        spec = RESEARCHERS[researcher_id]
    except KeyError as exc:
        raise ValueError(f"unknown researcher: {researcher_id}") from exc
    return IsolatedResearcher(
        spec,
        checkout=checkout,
        python=python,
        worker=worker,
        config=config,
        timeout_seconds=timeout_seconds,
    )


def _research_prompt(
    campaign: CampaignSpec,
    evidence: EvidenceSnapshot,
    sources: list[ResearchSource],
) -> str:
    payload = {
        "objective": campaign.objective,
        "primary_metric": campaign.primary_metric,
        "track": campaign.track,
        "allowed_knobs": sorted(campaign.allowed_knobs),
        "evidence_snapshot_id": evidence.snapshot_id,
        "evidence": [item.model_dump(mode="json") for item in evidence.items[:120]],
        "sources": [item.model_dump(mode="json") for item in sources[:80]],
    }
    return (
        "Research this OpenUI SLM objective using primary sources. Return a cited "
        "memo that distinguishes data, model, and infrastructure causes and proposes "
        "one falsifiable bounded experiment. Do not emit shell, code, patches, or RL "
        "instructions.\n\n"
        + json.dumps(payload, sort_keys=True)
    )[:120_000]


def _sources_from_urls(text: str, researcher_id: str) -> tuple[ResearchSource, ...]:
    import re

    urls = {
        match.rstrip(".,;:!?")
        for match in re.findall(r"https?://[^\s<>\"'{}\[\]()]+", text)
    }
    return tuple(
        ResearchSource(
            source_id=f"researcher-{hashlib.sha256(url.encode()).hexdigest()[:16]}",
            kind="researcher",
            title=url,
            uri=url,
            metadata={"researcher_id": researcher_id},
        )
        for url in sorted(urls)
    )


def _merge_sources(
    *groups: list[ResearchSource] | tuple[ResearchSource, ...],
) -> tuple[ResearchSource, ...]:
    return tuple({source.uri: source for group in groups for source in group}.values())


def _file_identity(prefix: str, path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        f"{prefix}_bytes": len(data),
        f"{prefix}_sha256": hashlib.sha256(data).hexdigest(),
    }
