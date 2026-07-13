"""Auto-generate annotation prompts that differ from fixture test/train seeds.

Prompts are built from compositional templates so they can later be promoted
into held-out / train data without overlapping curated fixtures.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from slm_training.dsl.schema import load_jsonl

DEFAULT_SEED_PATH = Path("fixtures/train_seeds.jsonl")
DEFAULT_TEST_SEED_PATH = Path("fixtures/test_seeds.jsonl")

# Kept for /api/examples chips only (not used as annotation prompt bases).
EXAMPLE_PROMPTS = [
    "Hero card with title and body",
    "Primary call to action button",
    "Two feature cards stacked vertically",
    "Text blurb above a button",
    "Horizontal row of two buttons",
    "Pricing card with subscribe button",
]

_AUDIENCES = (
    "a fintech onboarding flow",
    "a clinic patient portal",
    "a marketplace seller dashboard",
    "a podcast studio landing page",
    "an internal ops console",
    "a university course page",
    "a travel itinerary planner",
    "a local bakery storefront",
)

_GOALS = (
    "capture an email signup",
    "compare two plan tiers",
    "confirm a destructive action",
    "summarize weekly activity",
    "highlight three differentiators",
    "collect shipping details",
    "show empty-state recovery",
    "walk through a three-step setup",
)

_LAYOUTS = (
    "a vertical Stack of Card + Button",
    "a row of two Cards under a TextContent header",
    "a Form with FormControl inputs and a submit Button",
    "Tabs with two TabItem panels of TextContent",
    "a Callout above a Card with CardHeader",
    "a Modal containing TextContent and Buttons",
    "a CardHeader title/subtitle over body TextContent",
    "an ImageBlock with caption TextContent and CTA Button",
)

_CONSTRAINTS = (
    "Use only placeholder strings for user-facing copy.",
    "Keep the tree shallow (at most one nested Stack).",
    "Prefer column layout; no absolute positioning.",
    "Include a clear primary action.",
    "Avoid decorative style literals; structure only.",
    "Make every leaf a named statement with placeholders.",
)

_TEMPLATES = (
    "For {audience}, generate OpenUI that can {goal}. Prefer {layout}. {constraint}",
    "Write an OpenUI screen for {audience} whose job is to {goal}. Structure it as {layout}. {constraint}",
    "Compose a compact OpenUI layout for {audience} to {goal}, using {layout}. {constraint}",
    "Produce OpenUI for {audience} focused on {goal}. Layout hint: {layout}. {constraint}",
)


def _fixture_prompt_set(
    *paths: Path | str | None,
) -> set[str]:
    out: set[str] = set()
    for raw in paths:
        if raw is None:
            continue
        path = Path(raw)
        if not path.exists():
            continue
        try:
            for row in load_jsonl(path):
                text = (row.prompt or "").strip().lower()
                if text:
                    out.add(text)
        except Exception:  # noqa: BLE001
            continue
    for ex in EXAMPLE_PROMPTS:
        out.add(ex.strip().lower())
    return out


def compose_prompt(rng: random.Random) -> str:
    tpl = rng.choice(_TEMPLATES)
    return tpl.format(
        audience=rng.choice(_AUDIENCES),
        goal=rng.choice(_GOALS),
        layout=rng.choice(_LAYOUTS),
        constraint=rng.choice(_CONSTRAINTS),
    )


def load_prompt_bank(seed_path: Path | str | None = DEFAULT_SEED_PATH) -> list[str]:
    """Legacy bank for examples chips; annotation cursor uses compose_prompt."""
    prompts: list[str] = list(EXAMPLE_PROMPTS)
    path = Path(seed_path) if seed_path else None
    if path and path.exists():
        try:
            for row in load_jsonl(path):
                text = (row.prompt or "").strip()
                if text and text not in prompts:
                    prompts.append(text)
        except Exception:  # noqa: BLE001
            pass
    return prompts


def vary_prompt(prompt: str, *, salt: int = 0) -> str:
    """Backward-compatible helper — compositional prompts are preferred."""
    seed = int.from_bytes(
        hashlib.sha256((str(int(salt)) + "\0" + prompt).encode("utf-8")).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    return compose_prompt(rng)


class PromptCursor:
    """Session-scoped generator of novel annotation prompts."""

    def __init__(
        self,
        bank: list[str] | None = None,
        *,
        session_id: str | None = None,
        vary: bool = True,
        blocked: set[str] | None = None,
    ) -> None:
        self.bank = list(bank or [])  # unused for generation; kept for API compat
        self.session_id = session_id or "default"
        seed = int(hashlib.sha1(self.session_id.encode()).hexdigest()[:8], 16)
        self._rng = random.Random(seed)
        self._n = 0
        self.vary = vary
        self._blocked = blocked or _fixture_prompt_set(
            DEFAULT_SEED_PATH, DEFAULT_TEST_SEED_PATH
        )
        self._seen: set[str] = set()

    def next(self) -> str:
        self._n += 1
        # Mix session RNG with step so successive prompts diverge.
        step_rng = random.Random(self._rng.randrange(1 << 30) ^ (self._n * 0x9E3779B9))
        for attempt in range(48):
            candidate = compose_prompt(
                random.Random(step_rng.randrange(1 << 30) ^ (attempt * 0x85EBCA6B))
            )
            key = candidate.strip().lower()
            if key in self._blocked or key in self._seen:
                continue
            self._seen.add(key)
            return candidate
        # Fallback must still avoid the fixture-blocked set.
        for attempt in range(64):
            candidate = (
                f"{compose_prompt(step_rng)} (novel {self._n}.{attempt})"
            )
            key = candidate.strip().lower()
            if key in self._blocked or key in self._seen:
                continue
            self._seen.add(key)
            return candidate
        # Extremely unlikely: return a unique synthetic prompt.
        unique = f"Compose a novel interface layout #{self.session_id}:{self._n}"
        self._seen.add(unique.lower())
        return unique
