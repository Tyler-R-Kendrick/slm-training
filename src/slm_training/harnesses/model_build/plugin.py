"""Model plug-in protocol and stub implementation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from slm_training.dsl.schema import ExampleRecord


class ModelPlugin(Protocol):
    def forward(self, batch: list[ExampleRecord]) -> float:
        """Return a scalar training loss for the batch."""

    def generate(self, prompt: str, gold: ExampleRecord | None = None) -> str:
        """Generate OpenUI for a prompt (stub may use gold)."""

    def save(self, path: Path) -> None:
        ...

    def load(self, path: Path) -> None:
        ...


@dataclass
class StubModel:
    """Deterministic stub that memorizes gold OpenUI by prompt.

    Used to prove harness wiring without a real TwoTower implementation.
    """

    memory: dict[str, str] = field(default_factory=dict)
    noise_rate: float = 0.0
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def forward(self, batch: list[ExampleRecord]) -> float:
        # "Train" by storing prompt → openui mappings; loss decreases with coverage.
        for record in batch:
            self.memory[record.prompt] = record.openui
        # Pseudo-loss: inverse of memory size (never zero for logging).
        return 1.0 / (1.0 + float(len(self.memory)))

    def generate(self, prompt: str, gold: ExampleRecord | None = None) -> str:
        if self.noise_rate > 0 and self._rng.random() < self.noise_rate:
            return "root = Broken("  # intentionally invalid
        if prompt in self.memory:
            return self.memory[prompt]
        if gold is not None:
            return gold.openui
        return 'root = Text(text=:stub.missing)'

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "stub",
            "memory": self.memory,
            "noise_rate": self.noise_rate,
            "seed": self.seed,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def load(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.memory = dict(payload.get("memory") or {})
        self.noise_rate = float(payload.get("noise_rate") or 0.0)
        self.seed = int(payload.get("seed") or 0)
        self._rng = random.Random(self.seed)
