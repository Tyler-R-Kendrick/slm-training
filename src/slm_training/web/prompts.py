"""Prompt seed bank + light variation for auto-generated annotation prompts."""

from __future__ import annotations

import hashlib
import random
import re
from pathlib import Path

DEFAULT_SEED_PATH = Path("fixtures/train_seeds.jsonl")

EXAMPLE_PROMPTS = [
    "Hero card with title and body",
    "Primary call to action button",
    "Two feature cards stacked vertically",
    "Text blurb above a button",
    "Horizontal row of two buttons",
    "Pricing card with subscribe button",
]

_PREFIXES = (
    "",
    "Design ",
    "Build ",
    "Create ",
    "Generate ",
    "Lay out ",
)
_SUFFIXES = (
    "",
    " for a landing page",
    " with clear hierarchy",
    " for mobile",
    " using placeholders",
)
_SWAPS = (
    (r"\bhero\b", "hero"),
    (r"\bpricing\b", "pricing"),
    (r"\bfeature\b", "feature"),
    (r"\bbutton\b", "button"),
    (r"\bcard\b", "card"),
    (r"\bform\b", "form"),
)


def load_prompt_bank(seed_path: Path | str | None = DEFAULT_SEED_PATH) -> list[str]:
    prompts: list[str] = list(EXAMPLE_PROMPTS)
    path = Path(seed_path) if seed_path else None
    if path and path.exists():
        try:
            from slm_training.dsl.schema import load_jsonl

            for row in load_jsonl(path):
                text = (row.prompt or "").strip()
                if text and text not in prompts:
                    prompts.append(text)
        except Exception:  # noqa: BLE001
            pass
    return prompts


def vary_prompt(prompt: str, *, salt: int = 0) -> str:
    """Apply light deterministic variation without changing intent."""
    rng = random.Random((hash(prompt) ^ int(salt)) & 0xFFFFFFFF)
    text = prompt.strip()
    # Occasionally lowercase first letter after a prefix swap.
    prefix = rng.choice(_PREFIXES)
    suffix = rng.choice(_SUFFIXES)
    if prefix and text[:1].isupper() and not text.lower().startswith(prefix.lower().strip()):
        body = text[0].lower() + text[1:] if rng.random() < 0.5 else text
        text = f"{prefix}{body}".strip()
    if suffix and suffix.lower() not in text.lower():
        # Avoid double punctuation.
        text = text.rstrip(".") + suffix
    # Tiny synonym-ish noun swaps via case-preserving regex.
    for pattern, _canon in _SWAPS:
        if rng.random() < 0.25 and re.search(pattern, text, flags=re.I):
            alts = {
                "hero": ["hero", "banner", "splash"],
                "pricing": ["pricing", "plan", "subscription"],
                "feature": ["feature", "benefit", "capability"],
                "button": ["button", "CTA", "action"],
                "card": ["card", "panel", "tile"],
                "form": ["form", "signup form", "input form"],
            }
            # Extract key from pattern like \bhero\b
            key = pattern.strip(r"\b")
            choices = alts.get(key, [key])
            repl = rng.choice(choices)

            def _sub(m: re.Match[str], r: str = repl) -> str:
                src = m.group(0)
                if src.isupper():
                    return r.upper()
                if src[:1].isupper():
                    return r[:1].upper() + r[1:]
                return r

            text = re.sub(pattern, _sub, text, count=1, flags=re.I)
            break
    return re.sub(r"\s+", " ", text).strip()


class PromptCursor:
    """Session-scoped rotating prompt cursor with optional variation."""

    def __init__(
        self,
        bank: list[str] | None = None,
        *,
        session_id: str | None = None,
        vary: bool = True,
    ) -> None:
        self.bank = list(bank or load_prompt_bank()) or list(EXAMPLE_PROMPTS)
        self.session_id = session_id or "default"
        seed = int(hashlib.sha1(self.session_id.encode()).hexdigest()[:8], 16)
        self._rng = random.Random(seed)
        self._i = self._rng.randrange(len(self.bank)) if self.bank else 0
        self._n = 0
        self.vary = vary

    def next(self) -> str:
        if not self.bank:
            return "Hero card with title and body"
        base = self.bank[self._i % len(self.bank)]
        self._i += 1
        self._n += 1
        if not self.vary:
            return base
        return vary_prompt(base, salt=self._n ^ self._rng.randrange(1 << 20))
