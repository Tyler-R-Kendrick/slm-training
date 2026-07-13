"""OpenUI parse/validate/serialize — backed by official @openuidev/lang-core.

The historical hand-rolled grammar lived here; harnesses now call the Node
bridge in ``tools/openui_bridge`` which uses ``createParser`` / ``jsonToOpenUI``.
"""

from __future__ import annotations

from slm_training.dsl.lang_core import ParseError, Program, parse, serialize, validate

__all__ = ["ParseError", "Program", "parse", "serialize", "validate"]
