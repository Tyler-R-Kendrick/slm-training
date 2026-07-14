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
  Empty,
  ErrorNote,
} from "../components";

export function Data({ navigate: _navigate }: { navigate: (to: string) => void }) {
  const caps = useCaps();
  const [version, setVersion] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const train = usePoll<any>(version ? `/api/data/train?version=${version}` : "/api/data/train", 0);
  const test = usePoll<any>("/api/data/test", 0);

  const v = train.data?.version;
  const recs = usePoll<any>(v ? `/api/data/train/${v}/records?limit=40` : null, 0);
  const stats = train.data?.stats;

  return (
    <div>
      <div className="page-head">
        <h1 className="page-title">Training Data</h1>
        <p className="page-sub">
          Navigate, generate, and validate versioned corpora. Cold-start shows committed
          fixtures; generated corpora live under <span className="mono">outputs/train_data/</span>.
        </p>
      </div>

      <ErrorNote error={train.error} />

      {train.data?.versions?.length > 0 && (
        <div className="chip-row" style={{ marginBottom: "1rem" }}>
          {train.data.versions.map((vv: string) => (
            <span key={vv} className={`chip ${vv === v ? "active" : ""}`} onClick={() => setVersion(vv)}>{vv}</span>
          ))}
        </div>
      )}

      <Grid min="190px">
        <StatTile label="Records" value={train.data?.record_count ?? stats?.record_count ?? "—"} accent="moss" sub={v ? `train_data/${v}` : "no built corpus"} />
        <StatTile label="Collected" value={stats?.collected_count ?? "—"} />
        <StatTile label="Quality rejected" value={stats?.quality_rejected ?? "—"} accent={stats?.quality_rejected ? "failed" : undefined} />
        <StatTile label="Synthesizer" value={stats?.synthesizer ?? "—"} />
        <StatTile label="Errors" value={stats?.error_count ?? "—"} accent={stats?.error_count ? "failed" : undefined} />
      </Grid>

      <div className="two-col">
        <Card title="Corpus composition" right={<ProvenanceBadge provenance={train.data?.provenance} />}>
          {stats ? (
            <Bars
              data={[
                { label: "records", value: stats.record_count ?? 0 },
                { label: "collected", value: stats.collected_count ?? 0 },
                { label: "quality rejected", value: stats.quality_rejected ?? 0 },
                { label: "reserved-test rejected", value: stats.structure_reserved_rejected ?? 0 },
              ]}
            />
          ) : (
            <Bars data={Object.entries(train.data?.fixture_counts ?? {}).map(([k, val]) => ({ label: k, value: val as number }))} />
          )}
        </Card>

        <Card title="Eval suites" right={<ProvenanceBadge provenance={test.data?.provenance} />}>
          <Bars data={Object.entries(test.data?.suites ?? {}).map(([k, val]) => ({ label: k, value: val as number }))} />
          <p className="hint" style={{ marginTop: "0.5rem" }}>Ship claims require full <span className="mono">rico_held</span> (n≥1500).</p>
        </Card>
      </div>

      <Card title="Records" right={<span className="hint">{recs.data?.count ?? 0} in corpus</span>}>
        {!v && <Empty>No built corpus. Generate one below.</Empty>}
        {v && (
          <DataTable
            columns={[
              { key: "id", label: "id" },
              { key: "split", label: "split" },
              { key: "source", label: "source" },
              { key: "prompt", label: "prompt" },
            ]}
            rows={recs.data?.records ?? []}
            render={{
              id: (r) => <span className="mono">{r.id}</span>,
              prompt: (r) => <span style={{ color: "var(--text-dim)" }}>{(r.prompt || "").slice(0, 90)}</span>,
            }}
          />
        )}
      </Card>

      <Card title="Generate">
        <div className="two-col">
          {jobDef(caps, "build_train_data") && (
            <JobLauncher
              jobDef={jobDef(caps, "build_train_data")!}
              execution={caps.execution}
              defaults={{ source: "fixture", version: "v1", synthesizer: "quality" }}
              onLaunched={setJobId}
            />
          )}
          {jobDef(caps, "build_test_data") && (
            <JobLauncher
              jobDef={jobDef(caps, "build_test_data")!}
              execution={caps.execution}
              defaults={{ source: "both", version: "v1", train_version: "v1" }}
              onLaunched={setJobId}
            />
          )}
        </div>
        <LogStream jobId={jobId} />
      </Card>
    </div>
  );
}
