import React, { useState } from "react";
import { usePoll } from "../api";
import { useCaps, jobDef } from "../caps";
import {
  Card,
  Grid,
  StatTile,
  StatusPill,
  DataTable,
  JobLauncher,
  LogStream,
  ProvenanceBadge,
  ErrorNote,
  fmt,
} from "../components";

const KINDS = ["research", "quality", "grammar", "perf", "phase"];

export function Experiments({ navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const [kind, setKind] = useState("research");
  const [jobId, setJobId] = useState<string | null>(null);
  const board = usePoll<any>(`/api/scoreboards/${kind}`, 30000);
  const flags = usePoll<any>("/api/experiment-flags", 15000);

  const results = board.data?.results ?? [];
  const metricColumns = board.data?.metric_columns ?? [];
  const passed = results.filter((r: any) => r.pass === true).length;

  function suiteMetric(row: any, suite: string, metric: string) {
    // The server normalizes dialects and applies the guarded legacy fallback
    // (tagged meaningful_source) — substituting parse_rate here would present
    // decoder-guaranteed syntax as meaningful quality.
    const values = row.suites?.[suite] ?? {};
    const v = values[metric];
    if (v === undefined || v === null) return "—";
    const legacy = metric === "meaningful_program_rate" && values.meaningful_source === "parse_rate_legacy";
    return `${fmt(v, 2)}${legacy ? "*" : ""}`;
  }

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Experiments</h1>
        <p className="page-sub">
          Ablation matrices — each row is one lever with a stable id. Cells are per-suite
          metrics against ship gates. Rows are evidence, not deployable models. Values
          marked * come from legacy pre-split boards where parse_rate stood in for meaningful.
        </p>
      </div>

      <div className="chip-row" style={{ marginBottom: "1rem" }}>
        {KINDS.map((k) => (
          <span key={k} className={`chip ${k === kind ? "active" : ""}`} onClick={() => setKind(k)}>{k}</span>
        ))}
      </div>

      <ErrorNote error={board.error} />
      <ErrorNote error={flags.error} />

      <Grid min="190px">
        <StatTile label="Feature flags" value={flags.data?.count ?? "—"} accent="moss" />
        <StatTile label="Boolean flags" value={flags.data?.boolean_count ?? "—"} />
        <StatTile label="Historical runs" value={flags.data?.history_runs ?? "—"} />
        <StatTile label="Registry" value={flags.data?.revision?.slice(0, 8) ?? "—"} />
      </Grid>

      <Card title="All feature flags" right={<ProvenanceBadge provenance={flags.data?.provenance} />}>
        <DataTable
          searchable
          searchPlaceholder="Search feature flags"
          columns={[
            { key: "label", label: "lever" },
            { key: "key", label: "OpenFeature key" },
            { key: "type", label: "type" },
            { key: "default", label: "default" },
          ]}
          rows={flags.data?.flags ?? []}
          render={{
            label: (r) => <span className="mono">{r.field}</span>,
            key: (r) => <span className="mono">{r.key}</span>,
            default: (r) => <span className="mono">{typeof r.default === "object" ? JSON.stringify(r.default) : fmt(r.default)}</span>,
          }}
        />
      </Card>

      <Grid min="190px">
        <StatTile label="Experiments" value={results.length} accent="moss" />
        <StatTile label="Passing" value={`${passed}/${results.length}`} accent={passed === results.length && results.length ? "passed" : undefined} />
        <StatTile label="Matrix" value={board.data?.meta?.matrix_set ?? board.data?.meta?.matrix ?? kind} />
        <StatTile label="Steps" value={board.data?.meta?.steps ?? "—"} />
      </Grid>

      <Card
        title={`${kind} matrix`}
        right={<><ProvenanceBadge provenance={board.data?.provenance} /> <span className="hint">{board.data?.reference}</span></>}
      >
        <DataTable
          columns={[
            { key: "id", label: "id" },
            { key: "date", label: "date" },
            { key: "description", label: "experiment" },
            { key: "pass", label: "gate" },
            ...metricColumns.map((c: any) => ({ key: c.key, label: c.label, align: "right" as const })),
            { key: "agentv", label: "AgentV", align: "right" },
            { key: "trace", label: "trace", align: "right" },
          ]}
          rows={results}
          render={{
            id: (r) => (
              <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id || r.id)}`)} title="open run detail">
                {r.id}
              </a>
            ),
            description: (r) => <span style={{ color: "var(--text-dim)" }}>{(r.description || "").slice(0, 70)}</span>,
            pass: (r) => (r.pass === undefined ? <span className="hint">—</span> : <StatusPill value={r.pass} label={r.pass ? "pass" : "fail"} />),
            ...Object.fromEntries(
              metricColumns.map((c: any) => [c.key, (r: any) => suiteMetric(r, c.suite, c.metric)]),
            ),
            agentv: (r) => r.agentv?.total === undefined ? "—" : `${r.agentv.passed ?? 0}/${r.agentv.total}`,
            trace: (r) => <span className="mono">{r.trace_id?.slice(0, 12) ?? "—"}</span>,
          }}
        />
      </Card>

      <Card title="Run a matrix">
        <div className="two-col">
          {jobDef(caps, "run_quality_matrix") && (
            <JobLauncher
              jobDef={jobDef(caps, "run_quality_matrix")!}
              execution={caps.execution}
              defaults={{ matrix: "v6", only: "E53", steps: 40, device: "cpu", context_backend: "scratch" }}
              onLaunched={setJobId}
            />
          )}
          {jobDef(caps, "run_grammar_matrix") && (
            <JobLauncher
              jobDef={jobDef(caps, "run_grammar_matrix")!}
              execution={caps.execution}
              defaults={{ only: "X0", steps: 40, device: "cpu" }}
              onLaunched={setJobId}
            />
          )}
        </div>
      </Card>

      <Card title="Dispatch a full GPU train" right={<span className="pill pill-warning">remote</span>}>
        <p className="hint" style={{ marginBottom: "0.7rem" }}>
          Heavy trains run on HF Jobs / a GPU pod — the dispatcher streams here, and durable
          checkpoints land in the HF bucket (see Overview → Remote dispatches). Use <span className="mono">dry_run</span> to
          preview without credentials.
        </p>
        <div className="two-col">
          {jobDef(caps, "hf_jobs_train") && (
            <JobLauncher
              jobDef={jobDef(caps, "hf_jobs_train")!}
              execution={caps.execution}
              defaults={{ run_id: "twotower_v1", steps: 200, dry_run: true }}
              onLaunched={setJobId}
            />
          )}
          {jobDef(caps, "remote_train") && (
            <JobLauncher
              jobDef={jobDef(caps, "remote_train")!}
              execution={caps.execution}
              defaults={{ host: "", run_id: "twotower_v1", steps: 200, dry_run: true }}
              onLaunched={setJobId}
            />
          )}
        </div>
      </Card>

      <LogStream jobId={jobId} />
    </div>
  );
}
