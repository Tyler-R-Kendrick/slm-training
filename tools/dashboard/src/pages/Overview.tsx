import React from "react";
import { usePoll } from "../api";
import {
  Card,
  DataTable,
  ErrorNote,
  Grid,
  ProvenanceBadge,
  StatTile,
  StatusPill,
  pct,
} from "../components";

const insightColumns = [
  { key: "finding", label: "Finding" },
  { key: "suggestion", label: "Suggested mitigation" },
];

export function Overview({ navigate }: { navigate: (to: string) => void }) {
  const { data, error } = usePoll<any>("/api/overview", 15000);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <div className="loading">Loading performance insights…</div>;

  const performance = data.performance ?? {};
  const references = performance.references ?? [];
  const comparisons = performance.comparisons ?? [];
  const insights = performance.insights ?? {};
  const stats = performance.stats ?? {};
  const cache = performance.cache ?? {};

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Performance insights</h1>
        <p className="page-sub">
          What to improve, what to carry forward, and signals that are easy to miss —
          anchored to the current model card and track champions.
        </p>
      </div>

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
            { key: "track", label: "Track" },
            { key: "run_id", label: "Run" },
            { key: "kind", label: "Kind" },
            { key: "location", label: "Artifact" },
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
            evaluation_status: (row) => <StatusPill value={row.evaluation_status} />,
            status: (row) => <span className="hint">{row.status || "—"}</span>,
          }}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>{performance.comparison_basis}</p>
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
            { key: "parse", label: "Parse", align: "right" },
            { key: "fidelity", label: "Fidelity", align: "right" },
            { key: "structure", label: "Structure", align: "right" },
            { key: "reward", label: "Reward", align: "right" },
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
            parse: (row) => pct(row.parse),
            fidelity: (row) => pct(row.fidelity),
            structure: (row) => pct(row.structure),
            reward: (row) => pct(row.reward),
          }}
        />
        <p className="hint" style={{ marginTop: "0.6rem" }}>
          Persisted {cache.generated_at ?? "—"}; regenerated only when the model-card checkpoint roster or a track champion changes.
        </p>
      </Card>
    </div>
  );
}
