# PR-head supervised science-lab diagnostic, 25 September 2026

[Machine-readable record](science-lab-pr-head-diagnostic-20260925-results.json).

First ordinary-supervisor invocation on commit `1e15899a1b71303d8de04e3ac90d6252c25e51aa` stopped before training. Campaign producer locked both manifests and paired design but omitted required `hypothesis_matrix_formed` event. Driver reported `FileNotFoundError: no formed hypothesis matrix`; supervisor recorded a repair wait. No model result, AgentV bundle, checkpoint, promotion, or ship evidence exists for this attempt. Its five-event preregistration and operation journal remain immutable; a new campaign must include the matrix before execution.

Planned diagnostic: CPU scratch TwoTower, one seed (`7301`), equal 65,826 parameters, six new optimizer updates per arm, six public smoke cases, paired masked-token denoising loss. No held-out or ship-gate claim applies. The first attempt measured none of these planned outcomes. Codex subscription was not invoked: no grant was configured in that invocation, and the generic driver operation lacks a safe reproducer recipe.

Source marker digest: `85c24c61ff4e67ee727e91cdd94420b56c7c4e026e40389df3d9ceabca07118a`. Plan artifact: `3647aa7f7e6a16b3c378244319e06be4a5b8a77602381db2acd1ab6ab360b221`. Retained log: `outputs/autonomy-integration-20260921/science-lab-r21-1e15899-prereg/supervisor-attempt-01.log`.
