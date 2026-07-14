import React, { useState } from "react";
import { usePoll } from "../api";
import { useCaps, jobDef } from "../caps";
import {
  Card,
  Grid,
  StatTile,
  DataTable,
  Bars,
  JobLauncher,
  LogStream,
  ProvenanceBadge,
  StatusPill,
  ErrorNote,
  fmt,
} from "../components";

export function Smoke({ navigate: _navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const [jobId, setJobId] = useState<string | null>(null);
  const perf = usePoll<any>("/api/scoreboards/perf", 0);
  const quality = usePoll<any>("/api/scoreboards/quality", 0);

  const perfRows = perf.data?.results ?? [];
  const first = perfRows[0] ?? {};
  const smokeRows = (quality.data?.results ?? []).map((r: any) => ({
    id: r.id,
    run_id: r.run_id,
    parse: r.suites?.smoke?.parse_rate,
    fidelity: r.suites?.smoke?.placeholder_fidelity,
    reward: r.suites?.smoke?.reward_score,
    n: r.suites?.smoke?.n,
  }));

  const phase = first.phase_summary ?? {};

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Smoke Runs</h1>
        <p className="page-sub">
          The fast canary — cheap wiring runs and the strictest-gated <span className="mono">smoke</span> suite.
          Latency, throughput, and phase telemetry at a glance. Smoke parse alone is a canary, not generalization.
        </p>
      </div>

      <ErrorNote error={perf.error} />

      <Grid min="180px">
        <StatTile label="Latency p50" value={first.latency_ms_p50 ? `${fmt(first.latency_ms_p50, 0)}ms` : "—"} accent="ember" />
        <StatTile label="Latency p95" value={first.latency_ms_p95 ? `${fmt(first.latency_ms_p95, 0)}ms` : "—"} />
        <StatTile label="Tokens/sec" value={fmt(first.tokens_per_sec, 1)} accent="moss" />
        <StatTile label="Wall sec" value={fmt(first.wall_sec, 2)} />
        <StatTile label="Perf runs" value={perfRows.length} />
      </Grid>

      <div className="two-col">
        <Card title="Phase breakdown (mean ms)" right={<span className="hint mono">{first.run_id || first.id || ""}</span>}>
          <Bars
            data={[
              { label: "denoiser", value: phase.denoiser_ms_mean ?? 0 },
              { label: "dfa sync", value: phase.dfa_sync_ms_mean ?? 0 },
              { label: "stream check", value: phase.stream_check_ms_mean ?? 0 },
            ]}
          />
          <p className="hint" style={{ marginTop: "0.5rem" }}>Where cycle time goes — target the tallest bar first.</p>
        </Card>

        <Card title="Perf matrix" right={<ProvenanceBadge provenance={perf.data?.provenance} />}>
          <DataTable
            columns={[
              { key: "id", label: "id" },
              { key: "latency_ms_p50", label: "p50 ms", align: "right" },
              { key: "tokens_per_sec", label: "tok/s", align: "right" },
              { key: "parse_rate", label: "parse", align: "right" },
            ]}
            rows={perfRows}
            render={{
              id: (r) => <span className="mono">{r.id}</span>,
              latency_ms_p50: (r) => fmt(r.latency_ms_p50, 0),
              tokens_per_sec: (r) => fmt(r.tokens_per_sec, 1),
              parse_rate: (r) => fmt(r.parse_rate, 2),
            }}
          />
        </Card>
      </div>

      <Card title="Smoke suite (canary)" right={<ProvenanceBadge provenance={quality.data?.provenance} />}>
        <DataTable
          columns={[
            { key: "id", label: "experiment" },
            { key: "parse", label: "parse", align: "right" },
            { key: "fidelity", label: "fidelity", align: "right" },
            { key: "reward", label: "reward", align: "right" },
            { key: "gate", label: "≥0.66 parse" },
          ]}
          rows={smokeRows}
          render={{
            id: (r) => <span className="mono">{r.id}</span>,
            parse: (r) => fmt(r.parse, 2),
            fidelity: (r) => fmt(r.fidelity, 2),
            reward: (r) => fmt(r.reward, 2),
            gate: (r) => (r.parse === undefined ? <span className="hint">—</span> : <StatusPill value={r.parse >= 0.66} label={r.parse >= 0.66 ? "pass" : "fail"} />),
          }}
        />
      </Card>

      <Card title="Launch a smoke / wiring run">
        <div className="two-col">
          {jobDef(caps, "run_perf_matrix") && (
            <JobLauncher jobDef={jobDef(caps, "run_perf_matrix")!} execution={caps.execution} defaults={{ device: "cpu", suite: "smoke" }} onLaunched={setJobId} />
          )}
          {jobDef(caps, "evaluate_model") && (
            <JobLauncher jobDef={jobDef(caps, "evaluate_model")!} execution={caps.execution} defaults={{ test_version: "v1", run_id: "playground_demo", suite: "smoke" }} onLaunched={setJobId} />
          )}
        </div>
        <LogStream jobId={jobId} />
      </Card>
    </div>
  );
}
