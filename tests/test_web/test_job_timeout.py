"""Hard-cap coverage for the deployed background-job runner."""

from __future__ import annotations

import asyncio
import sys

from slm_training.web import jobs


def test_job_hard_cap_interrupts_and_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(jobs, "INTERRUPT_AFTER_SECONDS", 0.05)
    monkeypatch.setattr(jobs, "KILL_GRACE_SECONDS", 0.05)
    registry = jobs.JobRegistry(tmp_path)
    job = jobs.Job(
        id="hard-cap",
        job_key="_sleep",
        argv=[sys.executable, "-c", "import time; time.sleep(60)"],
    )

    asyncio.run(registry._run(job))

    assert job.status == "failed"
    assert any("three-minute hard cap" in line for line in registry.tail(job.id))
