"""Built-in DSL pack instances (F1, SLM-34).

Each module here wires one DSL's existing component owners into a
:class:`slm_training.dsl.pack.DslPack`. OpenUI is the first pack; F2
(GraphQL), F3 (patterns DSL), and F4 (nomenclatures) register here when they
land.
"""

from slm_training.dsl.packs.openui import build_openui_pack

__all__ = ["build_openui_pack"]
