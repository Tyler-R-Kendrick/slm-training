"""Optional Trackio mirror; local campaign artifacts remain authoritative."""

from __future__ import annotations

from typing import Any


class TrackioSink:
    def __init__(self, *, project: str, run: str) -> None:
        self.project = project
        self.run = run
        self._started = False

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        try:
            import trackio
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install slm-training[research] for Trackio") from exc
        if not self._started:
            trackio.init(project=self.project, name=self.run)
            self._started = True
        if step is None:
            trackio.log(metrics)
        else:
            trackio.log(metrics, step=step)
