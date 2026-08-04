"""Build the checked, request-independent OpenUI completion artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.dsl.grammar.backends.types import GRAMMARS_DIR
from slm_training.dsl.grammar.fastpath.completion_artifact import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_MANIFEST_PATH,
    build_completion_artifact,
)
from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    grammar_path = GRAMMARS_DIR / "openui.lark"
    engine = OpenUIIncrementalEngine(grammar_path)
    manifest = build_completion_artifact(
        DSLNativeTokenizer.build(),
        engine._parser,
        grammar_path=grammar_path,
        artifact_path=args.artifact,
        manifest_path=args.manifest,
    )
    bound = manifest["state_min_terminals"]
    print(
        f"{args.artifact}: {manifest['lalr_state_count']} states, "
        f"{manifest['lalr_edge_count']} edges, "
        f"{manifest['lalr_rule_count']} rules, "
        f"{manifest['direct_token_count']} direct token rows, "
        f"acceptance-length lower bound max={bound['max']} "
        f"(trivial={bound['trivial']}, consumed by nothing)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
