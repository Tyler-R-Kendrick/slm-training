"""SLM-314 (LAR2-05): winner-take-all (WTA/MCL) training over multi-mode gold.

Problem: for many prompts there are several hard-valid canonical AST modes
(distinct verifier-accepted programs that all satisfy the prompt). Standard
single-gold CE picks ONE mode at the dataset level and gives zero gradient
toward the others (dataset-level mode collapse); duplicated-example
multi-gold CE averages the modes and can converge to an interpolated,
invalid hybrid. Winner-take-all / multiple-choice learning backpropagates
only the MINIMUM-loss mode per example, plus a small preregistered
coverage/floor term on the unused modes so they stay constrained instead of
drifting to zero probability.

This module provides:

- a deterministic, frozen fixture-scale multi-mode dataset (8 prompts x 2
  verifier-accepted canonical AST modes each), committed as a resource JSONL
  with per-mode alpha-invariant canonical AST fingerprints (SLM-299
  ``_canonical_key`` sha256 via SLM-312 ``ast_fingerprint``) plus a sha256
  freeze manifest — mode identity is NEVER the serialization string;
- :func:`mode_supervision_loss` — the existing ``training_loss`` terms
  computed deterministically per mode (dedicated seeded rng, model rng
  restored afterwards);
- :func:`winner_take_all_loss` — per example, backpropagate
  ``L_winner + floor_epsilon * mean(L_losers)`` with durable per-example
  selected-mode + per-mode loss telemetry;
- :func:`run_synthetic_two_mode` — a minimal two-mode analytic fixture
  proving single-gold CE collapses while WTA retains both modes (and
  multi-gold CE produces invalid hybrids), with deterministic assertions
  consumed by the tests and the CLI.

Set-FTPO machinery is NOT reused (previously rejected). Torch is imported
lazily inside the loss/synthetic functions so dataset-freeze tests stay
cheap.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm312_state_sources import (
    ast_fingerprint,
)
from slm_training.models.tree_edit_diffusion import (
    _is_valid,
    parse_statements,
)

EXPERIMENT_ID = "slm314-winner-take-all"
DATASET_SCHEMA = "slm314-multimode-dataset/v1"
DEFAULT_RESOURCE = Path("src/slm_training/resources/data/slm314_multimode/modes.jsonl")

# PREREGISTERED WTA floor weight (locked before any run; deviations are
# append-only and exploratory). The losing mode keeps probability mass
# ~ eps/(1+eps) per supervised factor at equilibrium instead of 0.
FLOOR_EPSILON = 0.1

# --------------------------------------------------------------------------- #
# Frozen multi-mode dataset
# --------------------------------------------------------------------------- #

# (prompt, [(mode_source, placeholders), ...]) — every mode must parse through
# the real validator and have a DISTINCT canonical AST fingerprint.
MULTIMODE_PROGRAMS: list[tuple[str, list[tuple[str, list[str]]]]] = [
    (
        "hero with cta",
        [
            (
                'root = Stack([t, b], "column")\n'
                't = TextContent(":hero.title")\n'
                'b = Button(":cta.label")',
                [":hero.title", ":cta.label"],
            ),
            (
                'root = Stack([c], "column")\n'
                'c = Card([t, b], "column")\n'
                't = TextContent(":hero.title")\n'
                'b = Button(":cta.label")',
                [":hero.title", ":cta.label"],
            ),
        ],
    ),
    (
        "simple text",
        [
            ('root = Stack([t], "column")\nt = TextContent(":body")', [":body"]),
            ('root = Stack([c], "column")\nc = Card([t], "column")\nt = TextContent(":body")', [":body"]),
        ],
    ),
    (
        "form with email and submit",
        [
            (
                'root = Form([i, s], "column")\n'
                'i = TextContent(":form.email")\n'
                's = Button(":form.submit")',
                [":form.email", ":form.submit"],
            ),
            (
                'root = Form([c], "column")\n'
                'c = Card([i, s], "column")\n'
                'i = TextContent(":form.email")\n'
                's = Button(":form.submit")',
                [":form.email", ":form.submit"],
            ),
        ],
    ),
    (
        "card with image",
        [
            (
                'root = Stack([c], "column")\n'
                'c = Card([im, t], "column")\n'
                'im = Image(":img.src")\n'
                't = TextContent(":img.caption")',
                [":img.src", ":img.caption"],
            ),
            (
                'root = Card([im, t], "column")\n'
                'im = Image(":img.src")\n'
                't = TextContent(":img.caption")',
                [":img.src", ":img.caption"],
            ),
        ],
    ),
    (
        "two texts",
        [
            (
                'root = Stack([a, b], "column")\n'
                'a = TextContent(":title")\n'
                'b = TextContent(":subtitle")',
                [":title", ":subtitle"],
            ),
            (
                'root = Stack([c], "column")\n'
                'c = Card([a, b], "column")\n'
                'a = TextContent(":title")\n'
                'b = TextContent(":subtitle")',
                [":title", ":subtitle"],
            ),
        ],
    ),
    (
        "button only",
        [
            ('root = Stack([b], "column")\nb = Button(":cta")', [":cta"]),
            ('root = Card([b], "column")\nb = Button(":cta")', [":cta"]),
        ],
    ),
    (
        "nested card",
        [
            (
                'root = Stack([c], "column")\n'
                'c = Card([t], "column")\n'
                't = TextContent(":card.body")',
                [":card.body"],
            ),
            (
                'root = Stack([c], "column")\n'
                'c = Card([inner], "column")\n'
                'inner = Card([t], "column")\n'
                't = TextContent(":card.body")',
                [":card.body"],
            ),
        ],
    ),
    (
        "form single field",
        [
            ('root = Form([i], "column")\ni = TextContent(":q")', [":q"]),
            (
                'root = Form([c], "column")\n'
                'c = Card([i], "column")\n'
                'i = TextContent(":q")',
                [":q"],
            ),
        ],
    ),
]


@dataclass(frozen=True)
class ModeV1:
    """One verifier-accepted canonical AST mode of a prompt."""

    mode_index: int
    source: str
    canonical_fingerprint: str  # alpha-invariant AST fingerprint (sha256)
    placeholders: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["placeholders"] = list(self.placeholders)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModeV1":
        return cls(
            mode_index=int(data["mode_index"]),
            source=str(data["source"]),
            canonical_fingerprint=str(data["canonical_fingerprint"]),
            placeholders=tuple(str(x) for x in data.get("placeholders") or ()),
        )


@dataclass(frozen=True)
class PromptGroupV1:
    """One prompt with >=2 verifier-accepted canonical AST modes."""

    group_id: str
    prompt: str
    modes: tuple[ModeV1, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "prompt": self.prompt,
            "modes": [m.to_dict() for m in self.modes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptGroupV1":
        return cls(
            group_id=str(data["group_id"]),
            prompt=str(data["prompt"]),
            modes=tuple(ModeV1.from_dict(m) for m in data.get("modes") or ()),
        )

    def records(self) -> list[ExampleRecord]:
        """One ExampleRecord per mode (identical prompt, distinct gold)."""
        return [
            ExampleRecord(
                id=f"{self.group_id}__m{mode.mode_index}",
                prompt=self.prompt,
                openui=mode.source,
                placeholders=list(mode.placeholders),
                split="train",
            )
            for mode in self.modes
        ]


def _fingerprint_source(source: str) -> str:
    """Alpha-invariant canonical AST fingerprint; raises when not valid."""
    if not _is_valid(source):
        raise ValueError(f"mode is not verifier-accepted: {source!r}")
    statements = parse_statements(source)
    if statements is None:
        raise ValueError(f"mode does not parse structurally: {source!r}")
    return ast_fingerprint(statements)


def build_multimode_dataset() -> list[PromptGroupV1]:
    """Deterministically construct the multi-mode dataset from the frozen spec.

    Fail-closed: every mode must be verifier-accepted through the real parser
    and every prompt must have >=2 DISTINCT canonical AST fingerprints.
    """
    groups: list[PromptGroupV1] = []
    for index, (prompt, mode_specs) in enumerate(MULTIMODE_PROGRAMS):
        modes: list[ModeV1] = []
        seen: set[str] = set()
        for mode_index, (source, placeholders) in enumerate(mode_specs):
            fp = _fingerprint_source(source)
            if fp in seen:
                raise ValueError(
                    f"prompt {prompt!r} mode {mode_index} duplicates a canonical "
                    "AST fingerprint — modes must be distinct canonical ASTs"
                )
            seen.add(fp)
            modes.append(
                ModeV1(
                    mode_index=mode_index,
                    source=source,
                    canonical_fingerprint=fp,
                    placeholders=tuple(placeholders),
                )
            )
        if len(modes) < 2:
            raise ValueError(f"prompt {prompt!r} has fewer than 2 modes")
        groups.append(
            PromptGroupV1(
                group_id=f"mm_{index:02d}", prompt=prompt, modes=tuple(modes)
            )
        )
    return groups


def _digest(obj: Any) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataset_manifest(groups: Sequence[PromptGroupV1]) -> dict[str, Any]:
    row_hashes = [_digest(g.to_dict()) for g in groups]
    return {
        "dataset_schema": DATASET_SCHEMA,
        "n_prompts": len(groups),
        "n_modes": sum(len(g.modes) for g in groups),
        "modes_per_prompt": sorted({len(g.modes) for g in groups}),
        "row_sha256": row_hashes,
        "manifest_sha256": _digest({"schema": DATASET_SCHEMA, "row_sha256": row_hashes}),
    }


class FreezeIntegrityError(RuntimeError):
    """A frozen multi-mode dataset row or manifest failed verification."""


def freeze_dataset(groups: Sequence[PromptGroupV1], path: Path) -> dict[str, Any]:
    """Write the dataset as JSONL plus a ``<path>.manifest.json`` sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for group in groups:
            fh.write(json.dumps(group.to_dict(), sort_keys=True) + "\n")
    manifest = dataset_manifest(groups)
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def read_frozen_dataset(path: Path) -> list[PromptGroupV1]:
    """Load the frozen dataset, verifying row hashes + manifest (fail-closed)."""
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups: list[PromptGroupV1] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                groups.append(PromptGroupV1.from_dict(json.loads(line)))
    if [_digest(g.to_dict()) for g in groups] != list(manifest.get("row_sha256") or []):
        raise FreezeIntegrityError(f"row hash mismatch against {manifest_path}")
    if dataset_manifest(groups)["manifest_sha256"] != manifest.get("manifest_sha256"):
        raise FreezeIntegrityError(f"manifest sha256 mismatch for {manifest_path}")
    # Re-verify acceptance + fingerprint identity (never trust strings alone).
    for group in groups:
        for mode in group.modes:
            if _fingerprint_source(mode.source) != mode.canonical_fingerprint:
                raise FreezeIntegrityError(
                    f"{group.group_id} mode {mode.mode_index} fingerprint does not "
                    "match its re-parsed canonical AST"
                )
    return groups


def single_gold_groups(groups: Sequence[PromptGroupV1]) -> list[PromptGroupV1]:
    """Deterministic single-gold view: per prompt, the mode with the
    lexicographically smallest canonical AST fingerprint."""
    reduced: list[PromptGroupV1] = []
    for group in groups:
        winner = min(group.modes, key=lambda m: m.canonical_fingerprint)
        reduced.append(
            PromptGroupV1(
                group_id=group.group_id, prompt=group.prompt, modes=(winner,)
            )
        )
    return reduced


# --------------------------------------------------------------------------- #
# Per-mode deterministic supervision + WTA loss
# --------------------------------------------------------------------------- #


def _seed_int(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def mode_supervision_loss(model: Any, record: ExampleRecord, *, seed: int) -> Any:
    """The existing ``training_loss`` terms for ONE gold mode, computed with a
    dedicated seeded rng (the model's own rng state is saved/restored) so the
    per-mode loss is deterministic and identical across arms."""
    saved = model._rng
    model._rng = random.Random(seed)
    import torch

    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            return model.training_loss([record])
    finally:
        model._rng = saved


def select_winner(losses: Sequence[float]) -> int:
    """Index of the minimum-loss mode; ties break to the lowest mode index
    (deterministic)."""
    if not losses:
        raise ValueError("no mode losses")
    best = 0
    for i, value in enumerate(losses):
        if value < losses[best]:
            best = i
    return best


def winner_take_all_loss(
    model: Any,
    group: PromptGroupV1,
    *,
    floor_epsilon: float = FLOOR_EPSILON,
    seed: int = 0,
    eval_mode: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """WTA/MCL loss for one prompt group: backpropagate the minimum-loss mode
    plus ``floor_epsilon * mean(losing-mode losses)`` (coverage/floor term).

    Returns ``(loss, telemetry)``; telemetry records the per-mode losses, the
    selected winner fingerprint, and the floor weight (durable evidence).
    With ``eval_mode=True`` no gradient-capable loss is required — the same
    computation runs under ``torch.no_grad()`` by the caller.
    """
    records = group.records()
    losses: list[Any] = []
    detached: list[float] = []
    for mode, record in zip(group.modes, records):
        loss = mode_supervision_loss(
            model,
            record,
            seed=_seed_int(
                "slm314", str(seed), group.group_id, mode.canonical_fingerprint
            ),
        )
        losses.append(loss)
        detached.append(float(loss.detach().cpu()))
    winner = select_winner(detached)
    losers = [x for i, x in enumerate(losses) if i != winner]
    total = losses[winner]
    if losers and floor_epsilon > 0:
        total = total + floor_epsilon * (sum(losers) / len(losers))
    telemetry = {
        "group_id": group.group_id,
        "floor_epsilon": floor_epsilon,
        "winner_index": winner,
        "winner_fingerprint": group.modes[winner].canonical_fingerprint,
        "per_mode": [
            {
                "mode_index": mode.mode_index,
                "canonical_fingerprint": mode.canonical_fingerprint,
                "loss": detached[i],
                "selected": i == winner,
            }
            for i, mode in enumerate(group.modes)
        ],
        "total_loss": float(total.detach().cpu()),
    }
    return total, telemetry


# --------------------------------------------------------------------------- #
# Synthetic two-mode collapse-vs-retain fixture (analytic, deterministic)
# --------------------------------------------------------------------------- #

# Two positions, two tokens each. Mode A = (0, 0) "Stack/column", mode
# B = (1, 1) "Card/row". Hybrids (0, 1) and (1, 0) are INVALID — the
# token-level average of the two modes.
SYNTHETIC_MODE_A = (0, 0)
SYNTHETIC_MODE_B = (1, 1)

# PREREGISTERED synthetic assertions (locked before any run):
# - single-gold CE collapses: p(mode B) < SYNTHETIC_COLLAPSE_MAX
# - WTA retains both: p(mode B) >= SYNTHETIC_RETAINED_MIN and p(mode A) > p(B)
# - multi-gold CE averages: invalid-hybrid probability >= SYNTHETIC_HYBRID_MIN
SYNTHETIC_COLLAPSE_MAX = 0.002
SYNTHETIC_RETAINED_MIN = 0.004
SYNTHETIC_HYBRID_MIN = 0.3


def run_synthetic_two_mode(
    *,
    steps: int = 400,
    lr: float = 0.2,
    floor_epsilon: float = FLOOR_EPSILON,
) -> dict[str, dict[str, float]]:
    """Train a 2-position x 2-token softmax under the three arms.

    Full-batch, zero-init logits, no sampling — fully deterministic. Returns
    per-arm sequence probabilities ``{p_mode_a, p_mode_b, p_invalid_hybrid}``.
    """
    import torch

    def mode_loss(logits: Any, mode: tuple[int, int]) -> Any:
        import torch.nn.functional as F

        return sum(
            F.cross_entropy(logits[pos : pos + 1], torch.tensor([tok]))
            for pos, tok in enumerate(mode)
        )

    def probs(logits: Any) -> dict[str, float]:
        p = torch.softmax(logits, dim=-1)
        pa = float(p[0, 0] * p[1, 0])
        pb = float(p[0, 1] * p[1, 1])
        hybrid = float(p[0, 0] * p[1, 1] + p[0, 1] * p[1, 0])
        return {"p_mode_a": pa, "p_mode_b": pb, "p_invalid_hybrid": hybrid}

    out: dict[str, dict[str, float]] = {}
    for arm in ("single_gold", "multi_gold", "wta"):
        torch.manual_seed(0)  # identical init across arms (zero anyway)
        logits = torch.zeros(2, 2, requires_grad=True)
        opt = torch.optim.SGD([logits], lr=lr)
        for _ in range(steps):
            if arm == "single_gold":
                loss = mode_loss(logits, SYNTHETIC_MODE_A)
            elif arm == "multi_gold":
                loss = 0.5 * (
                    mode_loss(logits, SYNTHETIC_MODE_A)
                    + mode_loss(logits, SYNTHETIC_MODE_B)
                )
            else:
                la = mode_loss(logits, SYNTHETIC_MODE_A)
                lb = mode_loss(logits, SYNTHETIC_MODE_B)
                winner, loser = (la, lb) if float(la) <= float(lb) else (lb, la)
                loss = winner + floor_epsilon * loser
            opt.zero_grad()
            loss.backward()
            opt.step()
        out[arm] = probs(logits.detach())
    return out


def synthetic_assertions(result: dict[str, dict[str, float]]) -> dict[str, bool]:
    """Deterministic collapse-vs-retain assertions on the synthetic fixture."""
    return {
        "single_gold_collapses": (
            result["single_gold"]["p_mode_b"] < SYNTHETIC_COLLAPSE_MAX
        ),
        "wta_retains_loser_mode": (
            result["wta"]["p_mode_b"] >= SYNTHETIC_RETAINED_MIN
        ),
        "wta_winner_dominant": (
            result["wta"]["p_mode_a"] > result["wta"]["p_mode_b"]
        ),
        "multi_gold_hybrid_invalid": (
            result["multi_gold"]["p_invalid_hybrid"] >= SYNTHETIC_HYBRID_MIN
        ),
    }
