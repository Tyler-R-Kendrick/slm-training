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

const KINDS = ["quality", "grammar", "perf", "phase"];

export function Experiments({ navigate: _navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const [kind, setKind] = useState("quality");
  const [jobId, setJobId] = useState<string | null>(null);
  const board = usePoll<any>(`/api/scoreboards/${kind}`, 0);

  const results = board.data?.results ?? [];
  const passed = results.filter((r: any) => r.pass === true).length;

  function suiteMetric(row: any, suite: string, metric: string) {
    const v = row.suites?.[suite]?.[metric];
    return v === undefined ? "—" : fmt(v, 2);
  }

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Experiments</h1>
        <p className="page-sub">
          Ablation matrices — each row is one lever with a stable id. Cells are per-suite
          metrics against ship gates. Rows are evidence, not deployable models.
        </p>
      </div>

      <div className="chip-row" style={{ marginBottom: "1rem" }}>
        {KINDS.map((k) => (
          <span key={k} className={`chip ${k === kind ? "active" : ""}`} onClick={() => setKind(k)}>{k}</span>
        ))}
      </div>

      <ErrorNote error={board.error} />

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
            { key: "description", label: "lever" },
            { key: "pass", label: "gate" },
            { key: "smoke", label: "smoke parse", align: "right" },
            { key: "held", label: "held_out parse", align: "right" },
            { key: "struct", label: "struct", align: "right" },
          ]}
          rows={results}
          render={{
            id: (r) => <span className="mono">{r.id}</span>,
            description: (r) => <span style={{ color: "var(--text-dim)" }}>{(r.description || "").slice(0, 70)}</span>,
            pass: (r) => (r.pass === undefined ? <span className="hint">—</span> : <StatusPill value={r.pass} label={r.pass ? "pass" : "fail"} />),
            smoke: (r) => suiteMetric(r, "smoke", "parse_rate"),
            held: (r) => suiteMetric(r, "held_out", "parse_rate"),
            struct: (r) => suiteMetric(r, "smoke", "structural_similarity"),
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
        <LogStream jobId={jobId} />
      </Card>
    </div>
  );
}
