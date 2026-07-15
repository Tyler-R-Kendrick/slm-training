import React from "react";
import { usePoll } from "../api";
import {
  Card,
  Grid,
  StatTile,
  StatusPill,
  GateMatrix,
  Timeline,
  Bars,
  DataTable,
  ProvenanceBadge,
  Empty,
  ErrorNote,
  fmt,
} from "../components";

export function RunDetail({ runId, navigate }: { runId: string; navigate: (to: string) => void }) {
  const { data, error } = usePoll<any>(`/api/runs/${encodeURIComponent(runId)}`, 0);

  if (error) return <ErrorNote error={error} />;
  if (!data) return <div className="loading">Loading run…</div>;

  const ts = data.train_summary ?? {};
  const track = ts.track ?? {};
  const sb = data.scoreboard ?? {};
  const telem = data.telemetry ?? ts.telemetry ?? {};
  const spans = telem.spans ?? {};
  const bucket = ts.checkpoint_bucket ?? {};
  const finalEval = ts.final_eval?.suites ?? sb.suites ?? {};

  const spanBars = Object.entries(spans)
    .map(([name, v]: [string, any]) => ({ label: name, value: Number(v?.pct ?? 0) }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  const evalRows = Object.entries(finalEval).map(([suite, m]: [string, any]) => ({
    suite,
    parse_rate: m?.parse_rate,
    structural_similarity: m?.structural_similarity,
    placeholder_fidelity: m?.placeholder_fidelity,
    reward_score: m?.reward_score,
    n: m?.n,
  }));

  return (
    <div>
      <div className="page-head">
        <a className="chip" onClick={() => navigate("/experiments")} style={{ marginBottom: "0.6rem", display: "inline-block" }}>← back</a>
        <h1 className="page-title" style={{ display: "flex", gap: "0.6rem", alignItems: "center" }}>
          <span className="mono">{runId}</span>
          <ProvenanceBadge provenance={data.provenance} />
        </h1>
        <p className="page-sub">{sb.description || ts.model || "Run detail"}</p>
      </div>

      <Grid min="180px">
        <StatTile label="Steps" value={fmt(ts.steps)} accent="moss" sub={ts.stopped_on ? `stop: ${ts.stopped_on}` : undefined} />
        <StatTile label="Last loss" value={fmt(ts.last_loss, 4)} />
        <StatTile label="Best weighted NLL" value={fmt(ts.best_weighted_nll, 4)} accent="ember" />
        <StatTile label="Best ship score" value={fmt(ts.best_ship_score, 3)} />
        <StatTile label="Records" value={fmt(ts.record_count ?? sb.n)} />
        <StatTile label="Gate" value={data.gates?.pass === undefined ? "—" : ""} sub={sb.matrix ? `${sb.matrix} matrix` : undefined}
          accent={data.gates?.pass ? "passed" : data.gates ? "failed" : undefined} />
      </Grid>

      <div className="two-col">
        <Card title="Lifecycle">
          <Timeline state={data.manifest?.lifecycle_state ?? (sb.pass ? "screened" : "running")} />
          <DataTable
            columns={[{ key: "k", label: "Track" }, { key: "v", label: "" }]}
            rows={[
              { k: "context backend", v: track.context_backend ?? "—" },
              { k: "trainable params", v: fmt(track.trainable_params) },
              { k: "frozen params", v: fmt(track.frozen_params) },
              { k: "tokens / param", v: fmt(track.tokens_per_trainable_param, 2) },
              { k: "output tokenizer", v: track.output_tokenizer ?? "—" },
              { k: "finished", v: ts.finished_at ?? "—" },
            ]}
            render={{ v: (r) => <span className="mono" style={{ fontSize: "0.76rem" }}>{r.v}</span> }}
          />
          {bucket.bucket_url && (
            <a className="chip active" href={bucket.bucket_url} target="_blank" rel="noreferrer" style={{ marginTop: "0.6rem", display: "inline-block" }}>
              durable checkpoint ↗
            </a>
          )}
        </Card>

        <Card title="Ship gates" right={data.gates && <StatusPill value={!!data.gates.pass} label={data.gates.pass ? "pass" : "fail"} />}>
          {data.gates ? <GateMatrix gate={data.gates} /> : <Empty>No gate evidence for this run.</Empty>}
        </Card>
      </div>

      <div className="two-col">
        <Card title="Telemetry (share of cycle)">
          {spanBars.length ? <Bars data={spanBars} /> : <Empty>No telemetry for this run.</Empty>}
        </Card>
        <Card title="Eval suites">
          {evalRows.length ? (
            <DataTable
              columns={[
                { key: "suite", label: "suite" },
                { key: "parse_rate", label: "parse", align: "right" },
                { key: "structural_similarity", label: "struct", align: "right" },
                { key: "placeholder_fidelity", label: "fidelity", align: "right" },
                { key: "reward_score", label: "reward", align: "right" },
                { key: "n", label: "n", align: "right" },
              ]}
              rows={evalRows}
              render={{
                suite: (r) => <span className="mono">{r.suite}</span>,
                parse_rate: (r) => fmt(r.parse_rate, 2),
                structural_similarity: (r) => fmt(r.structural_similarity, 2),
                placeholder_fidelity: (r) => fmt(r.placeholder_fidelity, 2),
                reward_score: (r) => fmt(r.reward_score, 2),
              }}
            />
          ) : (
            <Empty>No eval scoreboard for this run.</Empty>
          )}
        </Card>
      </div>
    </div>
  );
}
