"""Fixtures for semantic plan tests."""

from __future__ import annotations

import pytest

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.dsl.pack import get_pack


SAMPLE_OPENUI = '''root = Stack([card, ok], "column")
card = Card([hello, world])
hello = TextContent(":hello.text")
world = TextContent(":world.text")
ok = Button(":ok.label")'''

SINGLE_OPENUI = 'root = TextContent(":copy.value")'


@pytest.fixture
def pack():
    return get_pack("openui")


@pytest.fixture
def sample_spec() -> ProgramSpec:
    return ProgramSpec.from_openui(
        id="sample",
        openui=SAMPLE_OPENUI,
        facts={},
        program_family_id="pf1",
        lineage_id="ln1",
        split_group_id="sg1",
    )


@pytest.fixture
def single_spec() -> ProgramSpec:
    return ProgramSpec.from_openui(
        id="single",
        openui=SINGLE_OPENUI,
        facts={},
        program_family_id="pf1",
        lineage_id="ln1",
        split_group_id="sg1",
    )
