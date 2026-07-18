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
import { metricLabel, smokeGate } from "../metrics";

function PhaseBreakdown({ runId, fallback }: { runId: string; fallback: any }) {
  const detail = usePoll<any>(runId ? `/api/runs/${encodeURIComponent(runId)}` : null, 0);
  const data = detail.data?.insights?.phases?.length
    ? detail.data.insights.phases
    : [
        { label: "denoiser", value: fallback.denoiser_ms_mean ?? 0, help: "Profile model forwards first; test AMP/compile, batching, or fewer decode steps, then rerun quality guardrails." },
        { label: "dfa sync", value: fallback.dfa_sync_ms_mean ?? 0, help: "Remove unnecessary host/device synchronization boundaries and measure again on the same device." },
        { label: "stream check", value: fallback.stream_check_ms_mean ?? 0, help: "Prefer incremental or chosen-token verification; keep final validation and parse guardrails enabled." },
      ];
  return <Bars data={data} />;
}

export function Smoke({ navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const [jobId, setJobId] = useState<string | null>(null);
  const perf = usePoll<any>("/api/scoreboards/perf", 0);
  const quality = usePoll<any>("/api/scoreboards/quality", 0);
  const gatePolicy = usePoll<any>("/api/gates/policy", 0);

  // Canary lever + threshold come from the live ship-gate policy — the pill
  // keeps meaning "clears the smoke gate" even when the lever changes.
  const gate = smokeGate(gatePolicy.data?.policy);
  const perfRows = perf.data?.results ?? [];
  const first = perfRows[0] ?? {};
  // The server normalizes dialects and tags the guarded legacy fallback; a
  // client-side parse_rate substitution would present decoder-guaranteed
  // syntax as the headline lever.
  const smokeRows = (quality.data?.results ?? []).map((r: any) => ({
    id: r.id,
    run_id: r.run_id,
    parse: r.suites?.smoke?.[gate.lever],
    parseLegacy: r.suites?.smoke?.meaningful_source === "parse_rate_legacy",
    parseCi: r.suites?.smoke?.[`${gate.lever}_ci95`],
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
          <PhaseBreakdown runId={first.run_id || first.id || ""} fallback={phase} />
          <p className="hint" style={{ marginTop: "0.5rem" }}>Where cycle time goes — target the tallest bar first.</p>
        </Card>

        <Card title="Perf matrix" right={<ProvenanceBadge provenance={perf.data?.provenance} />}>
          <DataTable
            columns={[
              { key: "id", label: "id" },
              { key: "latency_ms_p50", label: "p50 ms", align: "right", digits: 0, direction: "lower", help: "Median generation latency in milliseconds; lower is better." },
              { key: "tokens_per_sec", label: "tok/s", align: "right", digits: 1, direction: "higher", help: "Generated tokens per second; higher is better." },
              { key: "parse_rate", label: "parse", align: "right", digits: 2, direction: "higher", help: "Share of outputs that parse as valid OpenUI; higher is better." },
            ]}
            rows={perfRows}
            searchable
            searchPlaceholder="Search perf experiments"
            render={{ id: (r) => <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id || r.id)}`)}>{r.id}</a> }}
          />
        </Card>
      </div>

      <Card title="Smoke suite (canary)" right={<ProvenanceBadge provenance={quality.data?.provenance} />}>
        <DataTable
          columns={[
            { key: "id", label: "experiment" },
            { key: "parse", label: metricLabel(gate.lever), align: "right", digits: 2, help: "Headline smoke lever from the ship-gate policy. * = legacy board where parse_rate stood in for meaningful." },
            { key: "fidelity", label: "fidelity", align: "right", digits: 2, help: "Placeholder fidelity against the expected target." },
            { key: "reward", label: "reward", align: "right", digits: 2, help: "Aggregate smoke reward; higher is better." },
            { key: "gate", label: gate.label },
          ]}
          rows={smokeRows}
          searchable
          searchPlaceholder="Search smoke experiments"
          render={{
            id: (r) => <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id || r.id)}`)}>{r.id}</a>,
            parse: (r) =>
              `${fmt(r.parse, 2)}${r.parse != null && r.parseLegacy ? "*" : ""}${
                r.parse != null && Array.isArray(r.parseCi)
                  ? ` [${fmt(r.parseCi[0], 2)}, ${fmt(r.parseCi[1], 2)}]`
                  : ""
              }`,
            fidelity: (r) => fmt(r.fidelity, 2),
            reward: (r) => fmt(r.reward, 2),
            gate: (r) =>
              r.parse == null || gate.threshold == null ? (
                <span className="hint">—</span>
              ) : (
                <StatusPill value={r.parse >= gate.threshold} label={r.parse >= gate.threshold ? "pass" : "fail"} />
              ),
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
