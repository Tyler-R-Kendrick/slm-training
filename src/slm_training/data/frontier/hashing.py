"""Content hash binding a frozen frontier artifact to the exact gold it describes."""

from __future__ import annotations

import hashlib

from slm_training.data.leakage import norm_text
from slm_training.data.structure import strip_style_literals


def gold_content_hash(openui: str, prompt: str = "") -> str:
    """Stable 16-hex hash of a gold's (style-stripped openui + prompt).

    Frozen artifacts are keyed by this hash, so if a gold changes the skill's
    artifact filename no longer resolves and the stale artifact is dropped until
    the skill regenerates it.
    """
    payload = norm_text(strip_style_literals(openui or "")) + "\n" + norm_text(prompt or "")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
