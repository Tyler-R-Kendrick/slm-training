import React from "react";
import { usePoll } from "../api";
import {
  Card,
  DataTable,
  DispatchLines,
  ErrorNote,
  Grid,
  HeroStrip,
  JobLines,
  JobsBadgeView,
  OtelBadgesView,
  OtelRunLines,
  ProvenanceBadge,
  StatTile,
  StatusPill,
  pct,
} from "../components";
import { useHero } from "../hero";

const insightColumns = [
  { key: "finding", label: "Finding" },
  { key: "suggestion", label: "Suggested mitigation" },
];

export function Overview({ navigate }: { navigate: (to: string) => void }) {
  const { data, error } = usePoll<any>("/api/overview", 15000);
  const hero = useHero(15000);
  const { data: jobsRaw } = usePoll<any>("/api/jobs", 10000);
  const { data: dispRaw } = usePoll<any>("/api/dispatches", 30000);
  const { data: otelRaw } = usePoll<any>("/api/otel/runs", 10000);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <div className="loading">Loading mission control…</div>;

  const performance = data.performance ?? {};
  const references = performance.references ?? [];
  const comparisons = performance.comparisons ?? [];
  const insights = performance.insights ?? {};
  const stats = performance.stats ?? {};
  const cache = performance.cache ?? {};
  const metricColumns = performance.metric_columns ?? [];

  const activeJobs = (jobsRaw?.jobs ?? []).filter((j: any) => ["running", "queued"].includes(j.status));
  const jobsData = {
    rows: activeJobs.map((j: any) => ({ id: j.id, job: j.job_key, status: j.status })),
    execution: !!jobsRaw?.execution,
  };
  const dispData = {
    rows: (dispRaw?.jobs ?? []).map((j: any) => ({ id: j.id, job: j.job_key, status: j.status, url: j.remote_url })),
    remotes: (dispRaw?.remotes ?? []).map((r: any) => ({ run_id: r.run_id, url: r.url })),
    hf_jobs: dispRaw?.hf_jobs,
    bucket: dispRaw?.bucket,
  };

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Mission control</h1>
        <p className="page-sub">
          Champion status, ship-gate verdict, and what's in flight — grounded in
          live-vs-committed evidence from the model card and track champions.
        </p>
      </div>

      <HeroStrip hero={hero} navigate={navigate} />

      <div className="two-col">
        <Card title="Live jobs" right={<JobsBadgeView data={jobsData} />}>
          <JobLines data={jobsData} />
        </Card>
        <Card
          title="Remote dispatches"
          right={
            <span className={`prov ${dispRaw?.provenance === "live" ? "prov-live" : "prov-committed"}`}>
              hf jobs / bucket
            </span>
          }
        >
          <DispatchLines data={dispData} navigate={navigate} />
        </Card>
      </div>

      <Card title="Active training runs" right={<OtelBadgesView data={otelRaw} />}>
        <OtelRunLines data={otelRaw} navigate={navigate} />
      </Card>

      <Card
        title="Current model card"
        right={
          <>
            <ProvenanceBadge provenance={performance.reference_provenance} />{" "}
            <button className="chip" onClick={() => navigate("/checkpoints")}>open checkpoints →</button>
          </>
        }
      >
        <DataTable
          columns={[
            { key: "role", label: "Reference role" },
            { key: "run_id", label: "Run" },
            { key: "architecture", label: "Architecture" },
            { key: "parameters", label: "Parameters", align: "right" },
            { key: "model_size", label: "Model size", align: "right" },
            { key: "throughput", label: "Throughput", align: "right" },
            { key: "meaningful", label: "Meaningful", align: "right" },
            { key: "structure", label: "Structure", align: "right" },
            { key: "evaluation_status", label: "Evaluation" },
            { key: "status", label: "Model-card status" },
          ]}
          rows={references}
          render={{
            run_id: (row) => (
              <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(row.run_id)}`)}>
                {row.run_id || "—"}
              </a>
            ),
            meaningful: (row) => pct(row.metrics?.meaningful_program_rate),
            structure: (row) => pct(row.metrics?.structural_similarity),
            evaluation_status: (row) => <StatusPill value={row.evaluation_status} />,
            status: (row) => <span className="hint">{row.status || "—"}</span>,
          }}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>{performance.comparison_basis}</p>
        <p className="hint">
          Resource figures marked ≈ are architecture-comparable estimates; exact figures come from local checkpoint,
          train-summary, or performance evidence. Throughput varies by hardware and decode settings.
        </p>
      </Card>

      <Grid min="210px">
        <StatTile label="Reference models" value={stats.reference_models ?? 0} sub="champions + latest checkpoint" accent="promoted" />
        <StatTile label="Experiments reviewed" value={stats.experiments ?? 0} sub="quality and grammar evidence" />
        <StatTile label="Gate clears" value={`${stats.passing ?? 0}/${stats.experiments ?? 0}`} sub="recorded experiment policy" accent="passed" />
        <StatTile label="Comparable deltas" value={stats.comparable ?? 0} sub={stats.comparable ? "against current reference" : "reference eval required"} accent={stats.comparable ? "moss" : "failed"} />
      </Grid>

      <Grid min="320px">
        <Card title="Improve next" right={<StatusPill value="failed" label="mitigate" />}>
          <DataTable columns={insightColumns} rows={insights.improvements ?? []} />
        </Card>
        <Card title="Carry forward" right={<StatusPill value="passed" label="preserve" />}>
          <DataTable columns={insightColumns} rows={insights.carry_forward ?? []} />
        </Card>
        <Card title="Easy-to-miss signals" right={<StatusPill value="running" label="investigate" />}>
          <DataTable columns={insightColumns} rows={insights.novel ?? []} />
        </Card>
      </Grid>

      <Card
        title="Experiment performance vs current reference"
        right={<span className={`prov prov-${cache.persisted ? "live" : "committed"}`}>{cache.persisted ? "insights persisted" : "session insight"}</span>}
      >
        <DataTable
          columns={[
            { key: "id", label: "Experiment" },
            { key: "run_id", label: "Run" },
            { key: "matrix", label: "Matrix" },
            { key: "gate_status", label: "Gate" },
            // Metric columns mirror the server's ship-gate policy, so lever
            // changes re-shape this table without a dashboard edit.
            ...metricColumns.map((c: any) => ({ key: c.key, label: c.label, align: "right" as const })),
            { key: "vs_reference", label: "Vs reference", align: "right" },
          ]}
          rows={comparisons}
          render={{
            run_id: (row) => (
              <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(row.run_id)}`)}>
                {row.run_id}
              </a>
            ),
            gate_status: (row) => <StatusPill value={row.gate_status} />,
            ...Object.fromEntries(
              metricColumns.map((c: any) => [c.key, (row: any) => pct(row.metrics?.[c.key])]),
            ),
          }}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          Persisted {cache.generated_at ?? "—"}; insight prose regenerates when the model-card roster, a track champion, or committed experiment boards change. Comparison rows always recompute.
        </p>
      </Card>
    </div>
  );
}
