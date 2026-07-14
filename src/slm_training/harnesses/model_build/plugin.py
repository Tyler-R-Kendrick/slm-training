"""Model plug-in protocol and stub implementation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.schema import ExampleRecord


class ModelPlugin(Protocol):
    def artifact_identity(self) -> dict[str, str]:
        """Return stable model/base/tokenizer identity fields."""

    def compatibility_fingerprint(self) -> str:
        """Hash architecture, tokenizer, base revision, and parameter shapes."""

    def forward(self, batch: list[ExampleRecord]) -> float:
        """Return a scalar training loss for the batch."""

    def generate(self, prompt: str, gold: ExampleRecord | None = None) -> str:
        """Generate OpenUI for a prompt."""

    def generate_batch_requests(self, requests: list[GenerationRequest]) -> list[str]:
        """Generate using production-available inputs only."""

    def generate_constrained(self, prompt: str, **kwargs: object) -> str:
        """Generate through the official OpenUI grammar constraint."""

    def export(self, path: Path, *, format: str) -> tuple[Path, ...]:
        """Export a deployment artifact."""

    def load_parent_weights(self, path: Path) -> None:
        """Load only parent model weights for a fresh branch optimizer."""

    def save(self, path: Path) -> None:
        ...

    def load(self, path: Path) -> None:
        ...


@dataclass
class StubModel:
    """Dict-memorization baseline (kept for ablations / no-torch smoke)."""

    memory: dict[str, str] = field(default_factory=dict)
    noise_rate: float = 0.0
    seed: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def forward(self, batch: list[ExampleRecord]) -> float:
        for record in batch:
            self.memory[record.prompt] = record.openui
        return 1.0 / (1.0 + float(len(self.memory)))

    def generate(self, prompt: str, gold: ExampleRecord | None = None) -> str:
        _ = gold  # never oracle-leak at eval
        if self.noise_rate > 0 and self._rng.random() < self.noise_rate:
            return "root = Broken("
        if prompt in self.memory:
            return self.memory[prompt]
        return 'root = Stack([missing])\nmissing = TextContent(":stub.missing")'

    def generate_batch(
        self,
        prompts: list[str],
        golds: list[ExampleRecord | None] | None = None,
        **_kwargs: object,
    ) -> list[str]:
        _ = golds
        return [self.generate(p) for p in prompts]

    def generate_batch_requests(self, requests: list[GenerationRequest]) -> list[str]:
        return [self.generate(request.prompt) for request in requests]

    def artifact_identity(self) -> dict[str, str]:
        from slm_training.lineage.records import content_sha

        return {
            "kind": "stub",
            "base_model_id": "stub",
            "base_model_revision": "local",
            "tokenizer_sha": content_sha("stub"),
        }

    def compatibility_fingerprint(self) -> str:
        from slm_training.lineage.records import content_sha

        return content_sha({"kind": "stub", "keys": sorted(self.memory)})

    def generate_constrained(self, prompt: str, **kwargs: object) -> str:
        from slm_training.dsl.parser import validate

        output = self.generate(prompt)
        validate(output)
        return output

    def export(self, path: Path, *, format: str = "json") -> tuple[Path, ...]:
        if format != "json":
            raise ValueError("stub export supports format='json' only")
        self.save(path)
        return (path,)

    def load_parent_weights(self, path: Path) -> None:
        self.load(path)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "kind": "stub",
            "memory": self.memory,
            "noise_rate": self.noise_rate,
            "seed": self.seed,
        }
        # Keep .pt extension but write JSON for stub
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def load(self, path: Path) -> None:
        # Support both JSON stub and accidental torch files
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        self.memory = dict(payload.get("memory") or {})
        self.noise_rate = float(payload.get("noise_rate") or 0.0)
        self.seed = int(payload.get("seed") or 0)
        self._rng = random.Random(self.seed)
