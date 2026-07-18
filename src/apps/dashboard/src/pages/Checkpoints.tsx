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

export function Checkpoints({ navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const roster = usePoll<any>("/api/checkpoints", 0);
  const quality = usePoll<any>("/api/scoreboards/quality", 0);
  const champions = usePoll<any>("/api/lineage/champions", 0);

  const [policy, setPolicy] = useState<Record<string, Record<string, number>> | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [gate, setGate] = useState<GatePayload | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  const runs = (quality.data?.results ?? []).filter((r: any) => r.suites);

  useEffect(() => {
    getJSON<any>("/api/gates/policy").then((d) => setPolicy(d.policy)).catch(() => {});
  }, []);
  useEffect(() => {
    if (!runId && runs.length) setRunId(runs[0].id);
  }, [quality.data]);

  const selected = runs.find((r: any) => r.id === runId);

  useEffect(() => {
    if (!policy || !selected) return;
    postJSON<GatePayload>("/api/gates/evaluate", { suites: selected.suites, thresholds: policy })
      .then(setGate)
      .catch(() => setGate(null));
  }, [policy, runId, quality.data]);

  const cmp = usePoll<any>(runId ? `/api/comparisons/metrics?candidate_run_id=${runId}` : null, 0);
  const dep = roster.data?.deployment ?? {};
  const twChampion = champions.data?.champions?.twotower;

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Checkpoints &amp; Promotion</h1>
        <p className="page-sub">
          Navigate the roster, tune <strong>configurable gates</strong> live, and promote through the
          real honest-ship + lineage checks. Gate math runs server-side even read-only.
        </p>
      </div>

      <ErrorNote error={roster.error} />

      <Grid min="200px">
        <StatTile label="Checkpoints" value={(roster.data?.checkpoints ?? []).length} accent="moss" />
        <StatTile label="Deployed" value={dep.selected ? "1" : "0"} accent={dep.selected ? "promoted" : undefined} sub={dep.selected?.track ?? "none selected"} />
        <StatTile label="TwoTower champion" value={twChampion ? "set" : "none"} accent={twChampion ? "promoted" : undefined} sub={twChampion?.run_id ?? ""} />
        <StatTile label="A/B comparisons" value={cmp.data?.total ?? 0} sub={cmp.data ? `${pct(cmp.data.win_rate)} win` : ""} />
      </Grid>

      <Card title="Roster" right={<ProvenanceBadge provenance="committed" />}>
        <DataTable
          columns={[
            { key: "role", label: "Role" },
            { key: "run_id", label: "Run" },
            { key: "architecture", label: "Architecture" },
            { key: "parameters", label: "Parameters", align: "right" },
            { key: "model_size", label: "Model size", align: "right" },
            { key: "throughput", label: "Throughput", align: "right" },
            { key: "status", label: "Status" },
          ]}
          rows={roster.data?.checkpoints ?? []}
          render={{
            run_id: (r) => <span className="mono">{r.run_id || "—"}</span>,
            status: (r) => <StatusPill value={r.status} label={(r.status || "—").slice(0, 26)} />,
          }}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          ≈ denotes a comparable architecture estimate; throughput depends on hardware and decode settings.
        </p>
      </Card>

      <Card
        title="Configurable ship gates"
        right={
          <select value={runId ?? ""} onChange={(e) => setRunId(e.target.value)} style={{ background: "var(--bg-2)", color: "var(--text)", border: "1px solid var(--border-strong)", borderRadius: "4px", padding: "0.3rem" }}>
            {runs.map((r: any) => <option key={r.id} value={r.id}>{r.id} · {r.run_id}</option>)}
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
              <div className="hint" style={{ marginBottom: "0.5rem" }}>Edit per-suite minimum thresholds — the matrix re-evaluates live via the same <span className="mono">evaluate_ship_gates</span> the ship pipeline uses.</div>
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
