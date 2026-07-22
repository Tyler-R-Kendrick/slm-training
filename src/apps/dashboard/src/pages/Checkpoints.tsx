import React, { useEffect, useState } from "react";
import { usePoll, getJSON, postJSON } from "../api";
import { useCaps, jobDef } from "../caps";
import {
  Card,
  Grid,
  StatTile,
  StatusPill,
  DataTable,
  Timeline,
  GateMatrix,
  ThresholdEditor,
  JobLauncher,
  LogStream,
  ProvenanceBadge,
  Empty,
  ErrorNote,
  pct,
  fmt,
  type GatePayload,
} from "../components";

function metricCell(row: any, key: string) {
  const v = row.metrics?.[key];
  return v === undefined || v === null ? "—" : pct(v);
}

export function Checkpoints({ navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const roster = usePoll<any>("/api/checkpoints", 30000);
  const quality = usePoll<any>("/api/scoreboards/quality", 30000);
  const champions = usePoll<any>("/api/lineage/champions", 30000);

  const [policy, setPolicy] = useState<Record<string, Record<string, number>> | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [gate, setGate] = useState<GatePayload | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const checkpoints = roster.data?.checkpoints ?? [];
  const runs = (quality.data?.results ?? []).filter((r: any) => r.suites);
  // Prefer roster run_ids that already carry linked suite metrics; fall back to
  // the quality board so the gate editor still works on cold deploys.
  const gateOptions = [
    ...checkpoints
      .filter((c: any) => c.suites && Object.keys(c.suites).length)
      .map((c: any) => ({
        id: c.experiment_id || c.run_id,
        run_id: c.run_id,
        suites: c.suites,
      })),
    ...runs.filter(
      (r: any) => !checkpoints.some((c: any) => c.run_id === r.run_id && c.suites && Object.keys(c.suites).length),
    ),
  ];

  useEffect(() => {
    getJSON<any>("/api/gates/policy").then((d) => setPolicy(d.policy)).catch(() => {});
  }, []);
  useEffect(() => {
    if (!runId && gateOptions.length) setRunId(gateOptions[0].run_id || gateOptions[0].id);
  }, [roster.data, quality.data]);

  const selected =
    gateOptions.find((r: any) => r.run_id === runId || r.id === runId) ||
    runs.find((r: any) => r.id === runId || r.run_id === runId);

  useEffect(() => {
    if (!policy || !selected?.suites) return;
    postJSON<GatePayload>("/api/gates/evaluate", { suites: selected.suites, thresholds: policy })
      .then(setGate)
      .catch(() => setGate(null));
  }, [policy, runId, roster.data, quality.data]);

  const cmp = usePoll<any>(runId ? `/api/comparisons/metrics?candidate_run_id=${encodeURIComponent(runId)}` : null, 15000);
  const dep = roster.data?.deployment ?? {};
  const twChampion = champions.data?.champions?.twotower;

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Checkpoints &amp; Promotion</h1>
        <p className="page-sub">
          Navigate the roster, tune <strong>configurable gates</strong> live, and promote through the
          real honest-ship + lineage checks. Gate math runs server-side even read-only. Each run links
          to its stored experiment metrics.
        </p>
      </div>

      <ErrorNote error={roster.error} />

      <Grid min="200px">
        <StatTile label="Checkpoints" value={checkpoints.length} accent="moss" />
        <StatTile label="Deployed" value={dep.selected ? "1" : "0"} accent={dep.selected ? "promoted" : undefined} sub={dep.selected?.track ?? "none selected"} />
        <StatTile label="TwoTower champion" value={twChampion ? "set" : "none"} accent={twChampion ? "promoted" : undefined} sub={twChampion?.run_id ?? ""} />
        <StatTile label="A/B comparisons" value={cmp.data?.total ?? 0} sub={cmp.data ? `${pct(cmp.data.win_rate)} win` : ""} />
      </Grid>

      <Card title="Roster" right={<ProvenanceBadge provenance={roster.data?.provenance} />}>
        <DataTable
          columns={[
            { key: "role", label: "Role" },
            { key: "run_id", label: "Run" },
            { key: "architecture", label: "Architecture" },
            { key: "parameters", label: "Parameters", align: "right" },
            { key: "model_size", label: "Model size", align: "right" },
            { key: "gate_pass", label: "Gate" },
            { key: "meaningful", label: "Meaningful", align: "right" },
            { key: "structure", label: "Structure", align: "right" },
            { key: "agentv", label: "AgentV", align: "right" },
            { key: "evaluation_status", label: "Eval" },
            { key: "status", label: "Status" },
          ]}
          rows={checkpoints}
          render={{
            run_id: (r) =>
              r.run_id ? (
                <a
                  className="mono runlink"
                  onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id)}`)}
                  title="open run detail with linked metrics"
                >
                  {r.run_id}
                </a>
              ) : (
                <span className="mono">—</span>
              ),
            gate_pass: (r) =>
              r.gate_pass === undefined || r.gate_pass === null ? (
                <span className="hint">—</span>
              ) : (
                <StatusPill value={!!r.gate_pass} label={r.gate_pass ? "pass" : "fail"} />
              ),
            meaningful: (r) => metricCell(r, "meaningful_program_rate"),
            structure: (r) => metricCell(r, "structural_similarity"),
            agentv: (r) =>
              r.agentv?.total === undefined ? "—" : `${r.agentv.passed ?? 0}/${r.agentv.total}`,
            evaluation_status: (r) => <StatusPill value={r.evaluation_status || "—"} />,
            status: (r) => <StatusPill value={r.status} label={(r.status || "—").slice(0, 26)} />,
          }}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          ≈ denotes a comparable architecture estimate; Meaningful / Structure / AgentV come from the
          linked experiment scoreboard for the same run id.
          {roster.data?.bucket?.ok
            ? ` HF bucket: ${roster.data.bucket.count ?? 0} remote run(s)${roster.data.bucket.updated_at ? `, updated ${roster.data.bucket.updated_at}` : ""}.`
            : roster.data?.bucket?.error
              ? ` Bucket inventory unavailable: ${roster.data.bucket.error}.`
              : ""}
        </p>
      </Card>

      <Card
        title="Configurable ship gates"
        right={
          <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value)} style={{ background: "var(--bg-2)", color: "var(--text)", border: "1px solid var(--border-strong)", borderRadius: "4px", padding: "0.3rem" }}>
            {gateOptions.map((r: any) => (
              <option key={`${r.id}-${r.run_id}`} value={r.run_id || r.id}>
                {(r.run_id || r.id)}{r.id && r.run_id && r.id !== r.run_id ? ` · ${r.id}` : ""}
              </option>
            ))}
          </select>
        }
      >
        {!selected && (
          <Empty ctaLabel="run a smoke suite →" onCta={() => navigate("/smoke")}>
            No experiment with suite metrics available.
          </Empty>
        )}
        {selected && (
          <div className="two-col">
            <div>
              <div className="hint" style={{ marginBottom: "0.5rem" }}>
                Edit per-suite minimum thresholds — the matrix re-evaluates live via the same{" "}
                <span className="mono">evaluate_ship_gates</span> the ship pipeline uses. Selected run:{" "}
                <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(selected.run_id || selected.id)}`)}>
                  {selected.run_id || selected.id}
                </a>
              </div>
              {policy && <ThresholdEditor policy={policy} onChange={setPolicy} />}
            </div>
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.5rem" }}>
                <span className="hint">Overall</span>
                {gate && <StatusPill value={!!gate.pass} label={gate.pass ? "GATES PASS" : "GATES FAIL"} />}
              </div>
              {gate ? <GateMatrix gate={gate} /> : <Empty>Evaluating…</Empty>}
              {gate?.failures && gate.failures.length > 0 && (
                <ul className="hint" style={{ marginTop: "0.5rem", paddingLeft: "1.1rem" }}>
                  {gate.failures.slice(0, 6).map((f, i) => <li key={i} className="mono">{f}</li>)}
                </ul>
              )}
            </div>
          </div>
        )}
      </Card>

      <div className="two-col">
        <Card title="Lifecycle">
          <Timeline state={twChampion ? "champion" : selected ? "screened" : "running"} />
          <p className="hint" style={{ marginTop: "0.5rem" }}>Lineage lifecycle: running → screened → validated → champion → deployed.</p>
        </Card>

        <Card title="Deployment gate (blinded A/B)">
          {cmp.data ? (
            <>
              <Grid min="120px">
                <StatTile label="Comparisons" value={cmp.data.total} sub="need ≥100" accent={cmp.data.checks?.min_comparisons ? "passed" : undefined} />
                <StatTile label="Win rate" value={pct(cmp.data.win_rate)} sub="need >55%" accent={cmp.data.checks?.win_rate_gt_55 ? "passed" : undefined} />
                <StatTile label="Wilson LB" value={fmt(cmp.data.wilson_lower_bound, 2)} sub="need >0.50" accent={cmp.data.checks?.wilson_gt_50 ? "passed" : undefined} />
              </Grid>
              <StatusPill value={!!cmp.data.deployment_ready} label={cmp.data.deployment_ready ? "deployment ready" : "not ready"} />
            </>
          ) : (
            <Empty>No comparisons for this candidate yet.</Empty>
          )}
        </Card>
      </div>

      <Card title="Promote / Deploy">
        <div className="hint" style={{ marginBottom: "0.7rem" }}>
          These run <span className="mono">model_cycle</span>, which re-checks <span className="mono">promotion_failures</span> /
          <span className="mono"> deployment_failures</span> server-side — the UI never bypasses the real gate.
        </div>
        {jobDef(caps, "model_cycle") ? (
          <div className="two-col">
            <JobLauncher jobDef={jobDef(caps, "model_cycle")!} execution={caps.execution} defaults={{ subcommand: "promote", run_id: selected?.run_id ?? "", track: "twotower" }} onLaunched={setJobId} />
            <JobLauncher jobDef={jobDef(caps, "model_cycle")!} execution={caps.execution} defaults={{ subcommand: "deploy", run_id: selected?.run_id ?? "", track: "twotower" }} onLaunched={setJobId} />
          </div>
        ) : (
          <Empty>Promotion actions require the local control plane.</Empty>
        )}
        <LogStream jobId={jobId} />
      </Card>
    </div>
  );
}
