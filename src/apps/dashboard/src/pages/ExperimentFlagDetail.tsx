import React from "react";
import { usePoll } from "../api";
import { Card, DataTable, Empty, ErrorNote, ProvenanceBadge, StatTile, Grid, fmt } from "../components";

function display(value: any) {
  return typeof value === "object" ? JSON.stringify(value) : fmt(value);
}

export function ExperimentFlagDetail({
  flagKey,
  navigate,
}: {
  flagKey: string;
  navigate: (to: string) => void;
}) {
  const { data, error } = usePoll<any>(`/api/experiment-flags/${encodeURIComponent(flagKey)}`, 15000);
  const detail = data;
  const comparisons = detail?.comparisons ?? [];
  const evidence = detail?.evidence ?? [];

  return (
    <div>
      <div className="page-head">
        <button className="chip" onClick={() => navigate("/experiments")} style={{ marginBottom: "0.6rem" }}>← all feature flags</button>
        <h1 className="page-title mono">{flagKey}</h1>
        <p className="page-sub">OpenFeature flag detail, implementation evidence, and recorded experiment outcomes.</p>
      </div>

      <ErrorNote error={error} />
      {!detail ? <div className="loading">Loading feature flag…</div> : <>
        <Grid min="190px">
          <StatTile label="Type" value={detail.type} accent="moss" />
          <StatTile label="Default" value={display(detail.default)} />
          <StatTile label="Recorded runs" value={evidence.length} />
          <StatTile label="Comparisons" value={comparisons.length} />
        </Grid>

        <Card title="What it controls" right={<ProvenanceBadge provenance={detail.provenance} />}>
          <p style={{ margin: 0 }}>{detail.description}</p>
        </Card>

        <Card title="Implementation">
          {detail.implementation?.length ? detail.implementation.map((location: any) => (
            <details key={`${location.path}:${location.line}`} style={{ marginBottom: "0.65rem" }}>
              <summary className="mono runlink">{location.path}:{location.line}</summary>
              <pre style={{ margin: "0.55rem 0 0", overflowX: "auto", whiteSpace: "pre-wrap" }}>{location.snippet}</pre>
            </details>
          )) : <Empty>No local implementation source is available in this deployment.</Empty>}
        </Card>

        <Card title="Outcome comparisons" right={<ProvenanceBadge provenance={detail.provenance} />}>
          <p className="hint" style={{ marginBottom: "0.7rem" }}>{detail.comparison_note}</p>
          <DataTable
            searchable
            searchPlaceholder="Search comparisons"
            columns={[
              { key: "baseline_run_id", label: "previous baseline" },
              { key: "baseline_value", label: "baseline value" },
              { key: "run_id", label: "enabled / changed run" },
              { key: "value", label: "flag value" },
              { key: "outcome_summary", label: "recorded outcome change" },
            ]}
            rows={comparisons}
            render={{
              baseline_run_id: (r) => <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(r.baseline_run_id)}`)}>{r.baseline_run_id}</a>,
              baseline_value: (r) => <span className="mono">{display(r.baseline_value)}</span>,
              run_id: (r) => <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id)}`)}>{r.run_id}</a>,
              value: (r) => <span className="mono">{display(r.value)}</span>,
              outcome_summary: (r) => <span className="mono">{r.outcome_summary}</span>,
            }}
          />
        </Card>

        <Card title="Recorded run values">
          <DataTable
            searchable
            searchPlaceholder="Search recorded runs"
            columns={[
              { key: "run_id", label: "run" },
              { key: "date", label: "date" },
              { key: "value", label: "flag value" },
              { key: "state", label: "state" },
              { key: "pass", label: "gate" },
            ]}
            rows={evidence}
            render={{
              run_id: (r) => <a className="mono runlink" onClick={() => navigate(`/runs/${encodeURIComponent(r.run_id)}`)}>{r.run_id}</a>,
              value: (r) => <span className="mono">{display(r.value)}</span>,
            }}
          />
        </Card>
      </>}
    </div>
  );
}
