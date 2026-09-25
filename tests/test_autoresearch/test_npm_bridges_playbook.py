from slm_training.autoresearch.heal.playbooks.npm_bridges import PLAYBOOK


def test_environment_repair_installs_every_verifier_bridge_and_agentv(tmp_path):
    for path in (
        "src/apps/openui_bridge",
        "src/apps/design_md_bridge",
        "src/apps/graphql_bridge",
        "",
    ):
        root = tmp_path / path
        root.mkdir(parents=True, exist_ok=True)
        (root / "package.json").write_text("{}")

    plan = PLAYBOOK.plan(
        {"kind": "repair_harness", "reason": "AgentV SDK unavailable"},
        cwd=tmp_path,
    )

    assert plan is not None
    assert [step.cwd for step in plan.steps] == [
        "src/apps/openui_bridge",
        "src/apps/design_md_bridge",
        "src/apps/graphql_bridge",
        "",
    ]
